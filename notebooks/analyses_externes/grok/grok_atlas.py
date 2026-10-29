import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


@dataclass
class TsallisConfig:
    q_grid: np.ndarray = field(default_factory=lambda: np.arange(0.5, 2.55, 0.05))
    eps: float = 1e-12
    top_k: int = 128          # pour approximation stable des distributions

def tsallis_entropy(p: torch.Tensor, q: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Calcule S_q(p) de manière numériquement stable.
    p : (batch, vocab) ou (vocab,) – doit être une distribution de probabilité.
    """
    p = p.clamp(min=eps)
    if abs(q - 1.0) < 1e-6:
        return -torch.sum(p * torch.log(p), dim=-1)
    else:
        return (1.0 - torch.sum(p ** q, dim=-1)) / (q - 1.0)

def tsallis_divergence(p: torch.Tensor, r: torch.Tensor, q: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Divergence de Csiszár-Tsallis D_q(p || r).
    """
    p = p.clamp(min=eps)
    r = r.clamp(min=eps)
    if abs(q - 1.0) < 1e-6:
        return torch.sum(p * (torch.log(p) - torch.log(r)), dim=-1)
    else:
        return (torch.sum(p ** q * r ** (1 - q), dim=-1) - 1.0) / (q - 1.0)

def compute_spectral_surface(
    logits: torch.Tensor,
    q_grid: np.ndarray,
    top_k: int = 128
) -> np.ndarray:
    """
    Retourne S(q) pour une grille de q à partir des logits.
    Approximation top-k pour stabilité et vitesse.
    """
    probs = F.softmax(logits, dim=-1)
    if top_k is not None and top_k < probs.shape[-1]:
        top_probs, _ = torch.topk(probs, k=top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        probs = top_probs

    surfaces = []
    for q in q_grid:
        s = tsallis_entropy(probs, float(q))
        surfaces.append(s.detach().cpu().numpy())
    return np.stack(surfaces, axis=-1)  # (..., n_q)


@dataclass
class AttractorSignature:
    attractor_id: str
    spectral_mean: np.ndarray          # (n_q, max_t)
    spectral_std: np.ndarray
    production_profile: np.ndarray
    centroid_layers: Dict[int, np.ndarray]
    entry_directions: List[np.ndarray]
    exit_directions: List[np.ndarray]
    stability_score: float
    critical_tokens: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class NonExtensiveAtlas:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        q_config: Optional[TsallisConfig] = None,
        layers_to_track: Optional[List[int]] = None
    ):
        self.model_name = model_name
        self.device = device
        self.q_config = q_config or TsallisConfig()
        
        print(f"[Atlas] Chargement de {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.layers_to_track = layers_to_track or [0, self.num_layers // 4, self.num_layers // 2, 
                                                   3 * self.num_layers // 4, self.num_layers - 1]
        
        self.critical_tokens: Dict[str, Dict] = {}
        self.attractors: Dict[str, AttractorSignature] = {}
        self.transition_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        print(f"[Atlas] Modèle prêt. Layers suivis : {self.layers_to_track}")

    @torch.no_grad()
    def get_next_token_distribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Retourne les logits du prochain token + residual streams des layers suivis.
        """
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        logits = outputs.logits[:, -1, :]  # (batch, vocab)
        
        hidden_states = {}
        for layer_idx in self.layers_to_track:
            # hidden_states[0] = embedding, hidden_states[i+1] = after layer i
            hidden_states[layer_idx] = outputs.hidden_states[layer_idx + 1][:, -1, :].detach()
        
        return logits, hidden_states

    def compute_trajectory_signature(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Génère une trajectoire et enregistre la surface S(q,t) + residual streams.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        spectral_series = []
        residual_series = {l: [] for l in self.layers_to_track}
        generated_ids = inputs["input_ids"]
        
        for t in range(max_new_tokens):
            logits, hiddens = self.get_next_token_distribution(generated_ids)
            
            # Surface spectrale
            surface = compute_spectral_surface(
                logits, self.q_config.q_grid, top_k=self.q_config.top_k
            )
            spectral_series.append(surface[0])  # (n_q,)
            
            for l in self.layers_to_track:
                residual_series[l].append(hiddens[l][0].cpu().numpy())
            
            # Sampling
            if temperature > 0 and do_sample:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        spectral_array = np.stack(spectral_series, axis=0)  # (T, n_q)
        production = np.diff(spectral_array, axis=0)        # (T-1, n_q)
        
        return {
            "spectral": spectral_array,
            "production": production,
            "residuals": {l: np.stack(v, axis=0) for l, v in residual_series.items()},
            "generated_text": self.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
            "token_ids": generated_ids[0].cpu().tolist()
        }

    def discover_critical_tokens(
        self,
        candidate_tokens: List[str],
        base_prompts: List[str],
        n_samples: int = 8,
        sensitivity_threshold: float = 0.35
    ):
        """
        Identifie les tokens qui produisent un basculement significatif de régime.
        """
        print("[Atlas] Découverte des tokens critiques...")
        for token in tqdm(candidate_tokens):
            deltas = []
            for prompt in base_prompts:
                # Trajectoire contrôle
                sig_ctrl = self.compute_trajectory_signature(prompt, max_new_tokens=32)
                # Trajectoire avec injection
                injected = prompt + " " + token
                sig_inj = self.compute_trajectory_signature(injected, max_new_tokens=32)
                
                # Mesure de sensibilité : max |S_q - S_1| différentiel
                s_ctrl = sig_ctrl["spectral"].mean(axis=0)
                s_inj = sig_inj["spectral"].mean(axis=0)
                # Index de q=1 approximatif
                q1_idx = np.argmin(np.abs(self.q_config.q_grid - 1.0))
                delta = np.max(np.abs((s_inj - s_inj[q1_idx]) - (s_ctrl - s_ctrl[q1_idx])))
                deltas.append(delta)
            
            mean_delta = float(np.mean(deltas))
            if mean_delta > sensitivity_threshold:
                self.critical_tokens[token] = {
                    "sensitivity": mean_delta,
                    "q_star": float(self.q_config.q_grid[np.argmax(np.abs(s_inj - s_ctrl))]),
                    "status": "critical"
                }
        
        print(f"[Atlas] {len(self.critical_tokens)} tokens critiques identifiés.")

    def build_attractors_from_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        min_cluster_size: int = 5
    ):
        """
        Clustering des trajectoires dans l'espace des signatures spectrales
        + extraction des centroïdes de residual streams.
        """
        from sklearn.preprocessing import StandardScaler
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan est requis : pip install hdbscan")
        
        # Feature = surface spectrale aplatie + production moyenne
        features = []
        for traj in trajectories:
            spec = traj["spectral"]
            prod = traj["production"].mean(axis=0) if len(traj["production"]) > 0 else np.zeros(spec.shape[1])
            feat = np.concatenate([spec.mean(axis=0), prod, spec.std(axis=0)])
            features.append(feat)
        
        X = StandardScaler().fit_transform(np.array(features))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X)
        
        for label in set(labels):
            if label == -1:
                continue
            members = [trajectories[i] for i, l in enumerate(labels) if l == label]
            
            # Signature spectrale moyenne
            specs = np.stack([m["spectral"] for m in members], axis=0)
            # Padding à la longueur max
            max_t = max(s.shape[0] for s in specs)
            padded = []
            for s in specs:
                pad = np.pad(s, ((0, max_t - s.shape[0]), (0, 0)), mode="edge")
                padded.append(pad)
            specs_p = np.stack(padded, axis=0)
            
            spectral_mean = specs_p.mean(axis=0).T  # (n_q, T)
            spectral_std = specs_p.std(axis=0).T
            
            # Centroïdes residual
            centroids = {}
            for l in self.layers_to_track:
                res = [m["residuals"][l].mean(axis=0) for m in members]
                centroids[l] = np.mean(res, axis=0)
            
            att_id = f"A{label:03d}"
            self.attractors[att_id] = AttractorSignature(
                attractor_id=att_id,
                spectral_mean=spectral_mean,
                spectral_std=spectral_std,
                production_profile=np.mean([m["production"].mean(axis=0) for m in members if len(m["production"]) > 0], axis=0),
                centroid_layers=centroids,
                entry_directions=[],
                exit_directions=[],
                stability_score=float(clusterer.probabilities_[labels == label].mean()),
                critical_tokens=[],
                metadata={"n_members": len(members)}
            )
        
        print(f"[Atlas] {len(self.attractors)} attracteurs construits.")

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Sérialisation
        data = {
            "model_name": self.model_name,
            "q_grid": self.q_config.q_grid.tolist(),
            "critical_tokens": self.critical_tokens,
            "attractors": {
                k: {
                    "attractor_id": v.attractor_id,
                    "spectral_mean": v.spectral_mean.tolist(),
                    "spectral_std": v.spectral_std.tolist(),
                    "production_profile": v.production_profile.tolist(),
                    "centroid_layers": {str(lk): lv.tolist() for lk, lv in v.centroid_layers.items()},
                    "stability_score": v.stability_score,
                    "critical_tokens": v.critical_tokens,
                    "metadata": v.metadata
                } for k, v in self.attractors.items()
            }
        }
        with open(path / "atlas.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Atlas] Sauvegardé dans {path}")

    def load(self, path: str):
        # Implémentation symétrique (omise pour brièveté, mais triviale)
        pass
def build_atlas_pipeline():
    atlas = NonExtensiveAtlas(
        model_name="Qwen/Qwen3-8B",
        device="cuda",
        torch_dtype=torch.bfloat16
    )
    
    # 1. Prompts de base diversifiés
    base_prompts = [
        "Explique la notion d'entropie en physique statistique.",
        "Écris une courte histoire sur un voyageur temporel.",
        "Résous : si x^2 - 5x + 6 = 0, quelles sont les solutions ?",
        "Décris l'architecture d'un transformeur.",
        "Quelle est la capitale de l'Australie ?"
    ]
    
    # 2. Candidats (à enrichir par GlitchMiner ou analyse d'embedding)
    candidate_tokens = [
        "SolidGoldMagikarp", "TheNitrome", "\ufffd", 
        # ajouter d'autres tokens suspects ou rares
    ]
    
    # 3. Découverte
    atlas.discover_critical_tokens(candidate_tokens, base_prompts, n_samples=6)
    
    # 4. Génération de trajectoires pour clustering
    all_trajs = []
    for prompt in base_prompts:
        for _ in range(12):
            all_trajs.append(atlas.compute_trajectory_signature(prompt, max_new_tokens=48))
        for tok in atlas.critical_tokens:
            all_trajs.append(atlas.compute_trajectory_signature(prompt + " " + tok, max_new_tokens=48))
    
    # 5. Construction des attracteurs
    atlas.build_attractors_from_trajectories(all_trajs, min_cluster_size=4)
    
    # 6. Sauvegarde
    atlas.save("./qwen3_8b_nonextensive_atlas")
    
    return atlas
class AtlasMonitor:
    def __init__(self, atlas: NonExtensiveAtlas):
        self.atlas = atlas
    
    def detect_regime(self, current_spectral: np.ndarray) -> Optional[str]:
        """
        Compare la surface courante aux signatures de l'atlas.
        Retourne l'ID de l'attracteur le plus proche ou None.
        """
        best_id, best_dist = None, float("inf")
        for att_id, att in self.atlas.attractors.items():
            # Distance simple sur la moyenne spectrale
            ref = att.spectral_mean.mean(axis=1)
            dist = np.linalg.norm(current_spectral.mean(axis=0) - ref)
            if dist < best_dist:
                best_dist = dist
                best_id = att_id
        return best_id if best_dist < 0.45 else None  # seuil à calibrer

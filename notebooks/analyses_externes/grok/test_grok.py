"""Exécution réelle du code de Grok sur un Qwen3 jouet entraîné localement (Hugging Face bloqué ici).
Seul le chargement du modèle est remplacé ; toutes les méthodes de Grok tournent telles quelles."""
import sys, os, types, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
torch.manual_seed(0); np.random.seed(0)
import grok_atlas as GA
from transformers import Qwen3Config, Qwen3ForCausalLM

ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-«»\nàâäçéèêëîïôöùûüÀÇÉÈÊ"
class Tok:
    eos_token_id = 1
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]
    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[2 + ALPHA.index(c) if c in ALPHA else 2 + ALPHA.index("?") for c in text]])
        return types.SimpleNamespace(**{"input_ids": ids}, to=lambda d: {"input_ids": ids.to(d)}) if False else _BE(ids)
    def decode(self, ids, skip_special_tokens=True):
        return "".join(ALPHA[i - 2] if 2 <= i < 2 + len(ALPHA) else "" for i in ids)
class _BE(dict):
    def __init__(self, ids): super().__init__(input_ids=ids)
    def to(self, d): return _BE(self["input_ids"].to(d))

# Qwen3 jouet entraîné par caractères sur du texte français (comme nos harnais)
corpus_txt = ("La ville de Lyon est située au confluent du Rhône et de la Saône. Le principe de conservation de l'énergie "
              "affirme que l'énergie totale reste constante. Pour préparer une pâte à crêpes, il faut d'abord de la farine. "
              "Les réseaux de neurones artificiels sont des modèles inspirés du cerveau. Le chat s'approcha de la fenêtre. ") * 20
tok = Tok()
corpus = np.array(tok(corpus_txt)["input_ids"][0])
cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, max_position_embeddings=512)
m = Qwen3ForCausalLM(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
rng = np.random.default_rng(1)
for _ in range(600):
    d = rng.integers(0, len(corpus) - 65, 32)
    x = torch.tensor(np.stack([corpus[i:i + 64] for i in d]))
    loss = m(input_ids=x, labels=x).loss; opt.zero_grad(); loss.backward(); opt.step()
print(f"[JOUET] Qwen3 4 couches entraîné, perte finale {loss.item():.2f}")
m.eval()

atlas = object.__new__(GA.NonExtensiveAtlas)
atlas.model_name, atlas.device, atlas.q_config = "jouet", "cpu", GA.TsallisConfig()
atlas.tokenizer, atlas.model = tok, m
atlas.num_layers = 4
atlas.layers_to_track = [0, 1, 2, 3]
atlas.critical_tokens, atlas.attractors = {}, {}
from collections import defaultdict
atlas.transition_graph = defaultdict(dict)

# T1. Troncature top-k sur un grand vocabulaire (loi de Zipf, V = 151 936 comme Qwen3)
V = 151936
r = np.arange(1, V + 1); p = r ** -1.1; p /= p.sum()
pt = torch.tensor(p, dtype=torch.float64)
lg = torch.log(pt)
for q in (0.5, 1.0, 1.5, 2.5):
    full = float(GA.tsallis_entropy(pt, q))
    tk = float(torch.tensor(GA.compute_spectral_surface(lg[None], np.array([q]), top_k=128))[0, 0])
    print(f"[T1] Zipf V={V} q={q}: S_q complète {full:8.3f} ; top-128 renormalisée {tk:8.3f} ; rapport {tk / full:.2f}")

# T2. Bruit de la détection des « jetons critiques » : témoin contre témoin (injection vide) et même jeton deux fois
prompts = ["La ville de Lyon est", "Le principe de conservation", "Pour préparer une pâte", "Les réseaux de neurones", "Le chat s'approcha"]
atlas.discover_critical_tokens([""], prompts, n_samples=6)
print(f"[T2] injection vide (aucun jeton ajouté) déclarée critique : {'' in atlas.critical_tokens} "
      f"{atlas.critical_tokens.get('', {})}")
atlas.critical_tokens = {}
atlas.discover_critical_tokens(["SolidGoldMagikarp"], prompts)
s1 = atlas.critical_tokens.get("SolidGoldMagikarp", {}).get("sensitivity")
atlas.critical_tokens = {}
atlas.discover_critical_tokens(["SolidGoldMagikarp"], prompts)
s2 = atlas.critical_tokens.get("SolidGoldMagikarp", {}).get("sensitivity")
print(f"[T2] même jeton, deux exécutions : sensibilité {s1} puis {s2} (aucune graine fixée)")

# T3. Les « attracteurs » trouvés par le regroupement : bassins ou simplement prompts ?
trajs, lab = [], []
for i, pr in enumerate(prompts):
    for _ in range(12):
        trajs.append(atlas.compute_trajectory_signature(pr, max_new_tokens=48)); lab.append(i)
atlas.attractors = {}
atlas.build_attractors_from_trajectories(trajs, min_cluster_size=4)
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.metrics import adjusted_rand_score
feats = [np.concatenate([t["spectral"].mean(0), t["production"].mean(0), t["spectral"].std(0)]) for t in trajs]
X = StandardScaler().fit_transform(np.array(feats))
labels = hdbscan.HDBSCAN(min_cluster_size=4).fit_predict(X)
print(f"[T3] {len(set(labels) - {-1})} « attracteurs », {np.mean(labels == -1):.0%} de bruit ; accord avec l'identité du prompt "
      f"(indice de Rand ajusté) : {adjusted_rand_score(lab, labels):.2f}")
sp = np.array([t["spectral"].mean(0) for t in trajs])
cc = np.corrcoef(sp.T)
print(f"[T3] corrélation moyenne entre les 41 colonnes S_q de la « surface » : {cc[np.triu_indices(41, 1)].mean():.3f} ; "
      f"part de variance du premier axe : {np.linalg.svd(sp - sp.mean(0), compute_uv=False)[0]**2 / (np.linalg.svd(sp - sp.mean(0), compute_uv=False)**2).sum():.3f}")
S = np.concatenate([t["spectral"] for t in trajs])
print(f"[T3] pas de génération où S_q croît avec q quelque part : {int((np.diff(S, axis=1) > 1e-9).any(1).sum())} sur {len(S)}")

# T4. Couche Navigation : le flux d'utilisation tel qu'écrit, puis le visualiseur 3D
import grok_navigation as GN, grok_navigator3d as G3
nav = GN.NavigationLayer(atlas, projection_dim=3, use_umap=True)
feature_series = []
for traj in trajs[:10]:
    pass  # (dans la pratique on appelle une méthode helper)  ← tel quel dans le code de Grok
try:
    nav.fit_projector(feature_series)
    print("[T4] fit_projector sur le flux tel qu'écrit : OK")
except Exception as e:
    print(f"[T4] fit_projector sur le flux tel qu'écrit : ÉCHEC {type(e).__name__}: {str(e)[:120]}")
mid = sorted(atlas.layers_to_track)[len(atlas.layers_to_track) // 2]
fs = [nav._compute_features(t["spectral"], t["production"], t["residuals"][mid], atlas.q_config.q_grid) for t in trajs]
nav.fit_projector(fs)
for i in range(3):
    nav.create_orbit(f"s{i}", trajs[i * 12], prompts[i])
od = nav.overlay_orbits(["s0", "s1", "s2"])
print(f"[T4] bassins de l'Atlas transmis au visualiseur : {len(od['atlas_basins'])} (code : « pass »)")
try:
    fig = G3.AtlasNavigator3D().render(od, save_html=os.path.join(os.path.dirname(__file__), "carte.html"))
    print("[T4] rendu 3D : OK")
except Exception as e:
    print(f"[T4] rendu 3D : ÉCHEC {type(e).__name__}: {str(e)[:160]}")
F6 = np.vstack(fs)
print(f"[T4] corrélations entre S_0.7, S_1.0, S_1.5 (3 des 6 coordonnées) : "
      f"{np.corrcoef(F6[:, :3].T)[np.triu_indices(3, 1)].round(3).tolist()}")

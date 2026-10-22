# %% [markdown]
# # SGZ — Contrôle de criticité par la température (v2)
#
# **Contexte (v1, 21 oct. 2026).** Le pic de la Delta Entropy de Tsallis (G1, q = 1,45) tombait à T = 0,9 ; 1,0 ; 1,0
# sur Qwen3 0.6B, 1.7B et 8B (V1 SOUTENU), mais sa position dépendait de q (q = 0,5 : 2,0 ; q = 2,0 : 0,6–0,9 ;
# IC95 disjoints), et le pic de la susceptibilité χ était à 1,2 partout. La v2 teste si ce pic est un point critique,
# un artefact de mesure, ou un « bord » qui dépend du contexte (thèse d'Erick, 22 oct. 2026).
#
# **Changements.** Génération de 512 jetons (au lieu de 128) pour un test de taille finie par troncature ; quatre
# volets ; bootstrap TwoNN corrigé (sous-échantillonnage à 80 % sans remise) ; mesures stockées en float32.
#
# ## Volets et critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# **W1 — Invariance en q.** Pour CHAQUE q ∈ {0,5 ; 1 ; 1,45 ; 2}, fraction des rééchantillonnages dont l'argmax de G1
# (base p₁) tombe dans [0,8 ; 1,2]. SOUTENU si ≥ 95 % pour tous les q ; RÉFUTÉ si ≤ 5 % pour au moins un q ;
# sinon INDÉTERMINÉ. (Le choix d'inclure les 4 q de la v1, sans en retirer, est fait d'avance.)
#
# **W2 — Taille finie.** χ_L(T) = L · Var_prompts(surprisal moyenne sur les L premiers jetons) / T², pour
# L ∈ {64, 128, 256, 512}. Exposant b(T) = pente de log χ_L contre log L. Pour des jetons sans corrélation, χ_L ne
# dépend pas de L (b ≈ 0) ; des corrélations à longue portée, signature critique, font croître χ_L (b > 0).
# Statistique : Δb = b(T*_χ) − max(b(0,6), b(1,8)), T*_χ = argmax de χ₅₁₂. SOUTENU si IC95(Δb) > 0 ;
# RÉFUTÉ si la borne haute de l'IC95 < 0,05 ; sinon INDÉTERMINÉ.
#
# **W3 — Témoin de dispersion (mélange temporel).** G1_mél = G1 calculé après permutation aléatoire de l'ordre des
# pas de chaque séquence : il ne garde que la dispersion des valeurs de S, pas leur dynamique. Si
# |T*(G1_mél) − T*(G1)| ≤ 0,1 dans ≥ 95 % des rééchantillonnages → « PIC EXPLIQUÉ PAR LA DISPERSION » ;
# si ≤ 5 % → « PIC DYNAMIQUE » ; sinon INDÉTERMINÉ. (q = 1,45, base p₁.)
#
# **W4 — Thèse « le bord dépend du contexte ».** Par prompt, T_c = centroïde de la courbe G1(T) (q = 1,45, base p₁ ;
# poids = G1 − min, intégration trapézoïdale sur la grille).
# - **W4a, individualité** : T_c calculé sur les blocs pairs (A) et impairs (B) de 32 jetons ; ρ de Spearman entre
#   prompts. SOUTENU si ρ > 0 et p < 0,01 (5000 permutations) ; RÉFUTÉ si l'IC95 de ρ est dans [−0,1 ; 0,1] ;
#   sinon INDÉTERMINÉ.
# - **W4b, dérive dans la génération** : T_c(jetons 384–511) − T_c(jetons 0–127), moyenne sur les prompts.
#   SOUTENU si l'IC95 exclut 0 ; RÉFUTÉ si l'IC95 est dans [−0,03 ; 0,03] ; sinon INDÉTERMINÉ.
# - **W4c, instabilité initiale** : ρ de Spearman entre T_c et l'instabilité initiale du contexte (S₁,₄₅ moyen
#   sur les 16 premiers pas à T = 1). Prédiction de la thèse : ρ < 0. SOUTENU si IC95 < 0 ; CONTREDIT si IC95 > 0 ;
#   sinon INDÉTERMINÉ.
#
# Verdicts NON INTERPRÉTABLES en SMOKE_TEST ; NON TESTABLES si le témoin positif échoue ou si le modèle ne charge pas.
# Plusieurs volets et 3 modèles : environ 6 % de faux verdict par test sous le nul (mesuré en v1), à garder en tête.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# - W1 : RÉFUTÉ (85 %), la v1 montre déjà q = 0,5 au bord.
# - W2 : SOUTENU 30 %, RÉFUTÉ 25 %, INDÉTERMINÉ 45 %.
# - W3 : « pic expliqué par la dispersion » 50 %, « dynamique » 20 %, INDÉTERMINÉ 30 %.
# - W4a : SOUTENU 55 %. W4b : SOUTENU 45 %. W4c : SOUTENU 35 %, CONTREDIT 15 %, INDÉTERMINÉ 50 %.
#
# ## Antériorité (de mémoire, non vérifiée ici)
#
# Nakaishi, Nishikawa & Hukushima 2024 (arXiv:2406.05335) : Tc ≈ 1, traité comme une propriété fixe du modèle.
# Criticalité auto-organisée (Bak, Tang & Wiesenfeld 1987) ; phases de Griffiths, région critique étendue
# (Moretti & Muñoz 2013). Je ne connais pas de test d'un Tc dépendant du contexte dans un LLM.
#
# ## Limites connues
#
# - Le test de taille porte sur la longueur générée, pas sur la taille du modèle ; la troncature réutilise les mêmes
#   trajectoires (les L sont emboîtés, donc corrélés).
# - W4a : les blocs A et B viennent des mêmes trajectoires ; une corrélation peut venir d'un niveau propre au prompt
#   (par exemple une boucle à toutes les T). C'est bien une dépendance au contexte, mais pas forcément « dynamique ».
# - EOS, jetons spéciaux et identifiants hors tokenizer interdits ; entropies calculées avant ce masquage.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : environ 2 h 30 (0.6B ~25 min, 1.7B ~35 min,
# 8B ~85 min) [estimation à partir de la v1]. Budget automatique : 5 h.
#
# **À lire en priorité dans le log** : `[TÉMOIN POSITIF]`, `[PROGRESSION]`, puis `[W1]` à `[W4c]` et `[SYNTHÈSE]`.

# %%
import gc
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)   # chaque ligne du log apparaît tout de suite sur Kaggle
except Exception:
    pass
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_criticite_v2" if os.path.isdir("/kaggle/working") else "sgz_criticite_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_GRID = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
Q_LIST = [0.5, 1.0, 1.45, 2.0]
Q_PRIMARY = 1.45
WINDOW = (0.8, 1.2)
CHI_HALF_WIDTH = 0.2
N_PROMPTS, GEN_LEN, N_MIN = 64, 512, 32
L_SIZES = [64, 128, 256, 512]
BLOCK = 32
BOOT_N, TWONN_POINTS, TWONN_BOOT = int(os.environ.get("BOOT_N", "1000")), 2000, 50
LAYER_FRAC = 0.5
SEED = 20261021
MAX_HOURS = float(os.environ.get("MAX_HOURS", "5"))   # budget global : les modèles suivants sont sautés au-delà
T_RUN0 = time.time()

DEFAULT_MODELS = [
    ("Qwen3-0.6B-Base", ["Qwen/Qwen3-0.6B-Base"]),
    ("Qwen3-1.7B-Base", ["Qwen/Qwen3-1.7B-Base"]),
    ("Qwen3-8B-Base", ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B"]),
]
if os.environ.get("MODEL_IDS"):
    DEFAULT_MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]

if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, N_MIN = 8, 32, 32
    L_SIZES, BLOCK = [8, 16, 32], 4
    BOOT_N, TWONN_POINTS, TWONN_BOOT = 50, 100, 5
    DEFAULT_MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]

if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator). "
                       "Sur CPU, ce notebook prendrait des dizaines d'heures.")
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T_GRID={T_GRID} Q_LIST={Q_LIST} q principal={Q_PRIMARY} fenêtre={WINDOW}")
print(f"[CONFIG] N_PROMPTS={N_PROMPTS} GEN_LEN={GEN_LEN} BOOT_N={BOOT_N} modèles={[m[0] for m in DEFAULT_MODELS]}")
if DEVICE == "cuda":
    for i in range(N_GPU):
        p = torch.cuda.get_device_properties(i)
        print(f"[CONFIG] GPU {i} : {p.name}, {p.total_memory / 2**30:.1f} Go")
import transformers
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# %%
PROMPTS = [
    "La ville de Lyon est située au confluent",
    "Le principe de conservation de l'énergie affirme que",
    "Hier soir, en rentrant du travail, Marie a découvert",
    "Pour préparer une pâte à crêpes, il faut d'abord",
    "L'histoire de la Révolution française commence",
    "Les réseaux de neurones artificiels sont des modèles",
    "Le chat s'approcha lentement de la fenêtre et",
    "Dans un article publié la semaine dernière, les chercheurs",
    "La photosynthèse est le processus par lequel",
    "Il était une fois, dans un petit village de montagne,",
    "Le théorème de Pythagore permet de calculer",
    "Selon les prévisions météorologiques, demain il",
    "Le marché boursier a connu une forte baisse",
    "Pour installer ce logiciel, ouvrez un terminal et",
    "La musique baroque se caractérise par",
    "Le fleuve Saint-Laurent traverse",
    "Mon grand-père me racontait souvent que",
    "Les volcans se forment lorsque",
    "La recette traditionnelle du pot-au-feu",
    "En informatique, une fonction récursive est",
    "Le Québec compte plusieurs régions, dont",
    "La lumière se propage dans le vide à",
    "Le roman commence par une scène où",
    "Les abeilles jouent un rôle essentiel dans",
    "Le conseil municipal a voté hier une résolution",
    "Une étoile à neutrons est un objet",
    "Le premier jour d'école, Lucas avait peur de",
    "Pour résoudre une équation du second degré,",
    "La cathédrale Notre-Dame de Paris a été",
    "Les marées sont causées principalement par",
    "Le joueur a marqué un but à la dernière minute",
    "La langue française comporte de nombreux mots",
    "The city of Boston is known for",
    "In physics, the second law of thermodynamics states that",
    "Yesterday afternoon, the committee decided to",
    "To bake a loaf of bread, you first need to",
    "The history of the Roman Empire begins with",
    "Machine learning models are trained by",
    "The dog ran across the field and",
    "According to a recent study published in Nature,",
    "Photosynthesis converts sunlight into",
    "Once upon a time, in a kingdom by the sea,",
    "The Pythagorean theorem relates the sides of",
    "The weather forecast for tomorrow predicts",
    "Stock markets around the world fell sharply after",
    "To install the package, run the following command",
    "Jazz music originated in the early twentieth century",
    "The Amazon River flows through",
    "My grandmother used to tell me stories about",
    "Earthquakes occur when tectonic plates",
    "A classic recipe for tomato soup",
    "In computer science, a hash table is",
    "Canada is divided into ten provinces, including",
    "The speed of sound in air depends on",
    "The novel opens with a scene in which",
    "Honeybees communicate with each other by",
    "The city council approved a new budget",
    "A black hole is a region of spacetime",
    "On her first day at the new job, Sarah",
    "To solve a system of linear equations,",
    "The Eiffel Tower was built for",
    "Ocean currents are driven mainly by",
    "The team won the championship after",
    "The English language borrowed many words from",
]
assert len(PROMPTS) >= N_PROMPTS
PROMPTS = PROMPTS[:N_PROMPTS]
print(f"[PROMPTS] {len(PROMPTS)} prompts (moitié français, moitié anglais en mode complet)")

# %%
# ---------- Mesures ----------
@torch.no_grad()
def entropies_from_logits(logits, qs):
    """Entropies du softmax complet (float64). q = 1 : Shannon (nats) ; sinon Tsallis S_q = (1 − Σ p^q)/(q − 1)."""
    logp = torch.log_softmax(logits.double(), dim=-1)
    p = logp.exp()
    out = {}
    for q in qs:
        if q == 1.0:
            out[q] = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
        else:
            out[q] = (1.0 - (p ** q).sum(-1)) / (q - 1.0)
    return out


def tsallis_max(V, q):
    return math.log(V) if q == 1.0 else (1.0 - V ** (1.0 - q)) / (q - 1.0)


def g1_per_prompt(S):          # S : (P, L)
    return np.abs(np.diff(S, axis=1)).mean(axis=1)


def g2_per_prompt(S):
    return S.var(axis=1)


def curve_from_per_prompt(per_prompt_by_T, idx=None):
    """per_prompt_by_T : (nT, P). Courbe = moyenne sur les prompts (rééchantillonnés si idx)."""
    a = per_prompt_by_T if idx is None else per_prompt_by_T[:, idx]
    return a.mean(axis=1)


def g3_curve(meanS_by_T, idx=None):
    a = meanS_by_T if idx is None else meanS_by_T[:, idx]
    return a.shape[1] * a.var(axis=1)


def chi_curve(mean_surprisal_by_T, grid, L, idx=None):
    a = mean_surprisal_by_T if idx is None else mean_surprisal_by_T[:, idx]
    return L * a.var(axis=1) / np.asarray(grid) ** 2


def bootstrap_argmax(curve_fn, n_prompts, grid, B, rng):
    """Argmax de la courbe sur B rééchantillonnages appariés des prompts."""
    grid = np.asarray(grid)
    full = curve_fn(None)
    stars = np.empty(B)
    curves = np.empty((B, len(grid)))
    for b in range(B):
        idx = rng.integers(0, n_prompts, n_prompts)
        c = curve_fn(idx)
        curves[b] = c
        stars[b] = grid[int(np.argmax(c))]
    return full, stars, curves


def peak_verdict(full, stars, grid, window, interpretable=True):
    grid = np.asarray(grid)
    k = int(np.argmax(full))
    frac = float(np.mean((stars >= window[0] - 1e-9) & (stars <= window[1] + 1e-9)))
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif frac >= 0.95:
        v = "SOUTENU"
    elif frac <= 0.05:
        v = "RÉFUTÉ"
    else:
        v = "INDÉTERMINÉ"
    return {"T_star": float(grid[k]), "interior": bool(0 < k < len(grid) - 1), "frac_in_window": frac,
            "T_star_p2.5": float(np.percentile(stars, 2.5)), "T_star_p97.5": float(np.percentile(stars, 97.5)),
            "verdict": v}


@torch.no_grad()
def twonn(X, discard=0.1):
    """Dimension intrinsèque TwoNN (Facco et al. 2017) : régression par l'origine de −log(1 − F(μ)) sur log μ,
    μ = r2/r1, sur les (1 − discard) plus petits rapports. Les points dupliqués (r1 = 0) sont retirés et comptés."""
    X = X.float()
    n_all = X.shape[0]
    X = torch.unique(X, dim=0)          # doublons exacts retirés avant les distances (cdist float32 ne donne pas 0)
    dup = float(1.0 - X.shape[0] / max(n_all, 1))
    if X.shape[0] < 12:
        return float("nan"), dup
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    r, _ = torch.topk(D, 2, dim=1, largest=False)
    r1, r2 = r[:, 0], r[:, 1]
    ok = r1 > 1e-4 * r1.median()
    mu = (r2[ok] / r1[ok]).cpu().numpy()
    if mu.size < 10:
        return float("nan"), dup
    mu = np.sort(mu)
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = max(10, int(n * (1 - discard)))
    x, y = np.log(mu[:keep]), -np.log(1.0 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x)), dup


def gumbel_noise(V, seeds, device):
    """Bruit de Gumbel par ligne, graine explicite par ligne (mêmes nombres pour toutes les T d'un prompt)."""
    g = torch.Generator(device=device)
    rows = []
    for s in seeds:
        g.manual_seed(int(s))
        rows.append(torch.rand(V, generator=g, device=device, dtype=torch.float32))
    u = torch.stack(rows).clamp_(1e-10, 1 - 1e-7)
    return -torch.log(-torch.log(u))

# %%
# ---------- Témoin positif (synthétique) et contrôles de calcul ----------
rng_pc = np.random.default_rng(SEED + 1)
grid_np = np.asarray(T_GRID)


def synthetic_per_prompt(T0, P=64, amp=1.0, width=0.15, noise=0.5, rng=rng_pc):
    base = np.zeros(len(T_GRID)) if T0 is None else amp * np.exp(-(grid_np - T0) ** 2 / (2 * width ** 2))
    return base[:, None] + noise * rng.normal(size=(len(T_GRID), P))


pc = {}
for name, T0, expected in [("pic à 1,0", 1.0, "SOUTENU"), ("pic à 1,6", 1.6, "RÉFUTÉ"), ("sans pic", None, "INDÉTERMINÉ")]:
    arr = synthetic_per_prompt(T0)
    full, stars, _ = bootstrap_argmax(lambda idx, a=arr: curve_from_per_prompt(a, idx), arr.shape[1], T_GRID, 300, rng_pc)
    r = peak_verdict(full, stars, T_GRID, WINDOW)
    pc[name] = (r["verdict"], expected)
    print(f"[TÉMOIN POSITIF] {name:<10} : verdict {r['verdict']:<12} attendu {expected:<12} "
          f"(T*={r['T_star']}, fraction dans la fenêtre {r['frac_in_window']:.2f})")

false_v = 0
N_NULL = 100
for i in range(N_NULL):
    arr = synthetic_per_prompt(None, rng=np.random.default_rng(SEED + 100 + i))
    full, stars, _ = bootstrap_argmax(lambda idx, a=arr: curve_from_per_prompt(a, idx), arr.shape[1], T_GRID, 200,
                                      np.random.default_rng(SEED + 1000 + i))
    false_v += peak_verdict(full, stars, T_GRID, WINDOW)["verdict"] in ("SOUTENU", "RÉFUTÉ")
print(f"[TÉMOIN POSITIF] faux verdicts sous le nul (sans pic) : {false_v}/{N_NULL}")
POSITIVE_CONTROL_OK = all(v == e for v, e in pc.values()) and false_v <= 0.1 * N_NULL
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : tous les verdicts passent en NON TESTABLE'}")

for V in (10, 1000):
    u = torch.zeros(1, V)
    one = torch.full((1, V), -1e4)
    one[0, 0] = 0.0
    eu, eo = entropies_from_logits(u, Q_LIST), entropies_from_logits(one, Q_LIST)
    for q in Q_LIST:
        assert abs(eu[q].item() - tsallis_max(V, q)) < 1e-9 and abs(eo[q].item()) < 1e-9, (V, q)
print("[CONTRÔLE] S_q exacts sur loi uniforme et loi certaine")

z = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
cnt = np.zeros(4)
for s in range(20000):
    cnt[int(torch.argmax(z / 0.7 + gumbel_noise(4, [s], "cpu")))] += 1
p_ref = torch.softmax(z / 0.7, -1).numpy()[0]
print(f"[CONTRÔLE] Gumbel-max : fréquences {np.round(cnt / cnt.sum(), 3)} contre softmax {np.round(p_ref, 3)}")
assert np.max(np.abs(cnt / cnt.sum() - p_ref)) < 0.015

# %%
# ---------- Chargement ----------
class SmokeTokenizer:
    """Tokenizer jouet pour le SMOKE_TEST (caractères → identifiants)."""
    def __init__(self, V):
        self.V, self.pad_token_id, self.eos_token_id = V, 0, 1
        self.all_special_ids = [0, 1]

    def __len__(self):
        return self.V

    def encode(self, s):
        return [2 + (ord(c) % (self.V - 2)) for c in s][:12]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def transformers_major():
    return int(transformers.__version__.split(".")[0])


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == "smoke-qwen3-random":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
        torch.manual_seed(SEED)
        return Qwen3ForCausalLM(cfg).eval().to(DEVICE), SmokeTokenizer(512), "fp32 aléatoire (smoke)"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    big = any(s in model_id for s in ("7B", "8B", "9B"))
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    elif not big:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": 0}, "fp16 1 GPU"
    elif N_GPU >= 2:
        kw[dkey], kw["device_map"] = torch.float16, "auto"
        kw["max_memory"] = {i: "13GiB" for i in range(N_GPU)}
        mode = f"fp16 sur {N_GPU} GPU"
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"], mode = {"": 0}, "nf4 1 GPU"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda" and "nf4" not in mode:
        assert next(model.parameters()).dtype == torch.float16, "le modèle n'est pas en fp16"
    return model, tok, mode


def find_layers(model):
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("couches introuvables")

# %%
# ---------- Génération et mesures ----------
@torch.no_grad()
def run_model(model, tok, label):
    layers = find_layers(model)
    L_mid = int(round(LAYER_FRAC * (len(layers) - 1)))
    emb_dev = model.get_input_embeddings().weight.device
    V = model.get_output_embeddings().weight.shape[0]
    forbid = set(tok.all_special_ids or []) | {tok.eos_token_id}
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    forbid |= set(gen_eos if isinstance(gen_eos, (list, tuple)) else ([gen_eos] if gen_eos is not None else []))
    forbid.discard(None)
    n_tok = len(tok)
    print(f"[MESURE] {label} : {len(layers)} couches, couche TwoNN {L_mid}, V(modèle)={V}, len(tokenizer)={n_tok}, "
          f"{len(forbid)} jetons spéciaux interdits + {max(0, V - n_tok)} identifiants hors tokenizer")

    enc = [tok.encode(p) for p in PROMPTS]
    P, Lg = len(enc), GEN_LEN
    batch = 16 if sum(p.numel() for p in model.parameters()) < 3e9 else 8
    rng_pts = np.random.default_rng(SEED + 7)
    n_pts = min(TWONN_POINTS, P * (Lg - 1))
    flat = rng_pts.choice(P * (Lg - 1), n_pts, replace=False)
    sel = {(int(f // (Lg - 1)), int(f % (Lg - 1))) for f in flat}   # (prompt, k) : hidden du jeton généré k

    cap = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h

    hk = layers[L_mid].register_forward_hook(hook)
    res = {"S_p1": {q: np.zeros((len(T_GRID), P, Lg)) for q in Q_LIST},
           "S_pT": {q: np.zeros((len(T_GRID), P, Lg)) for q in Q_LIST},
           "S_frozen": {q: np.zeros((len(T_GRID), P, Lg)) for q in Q_LIST},
           "surprisal": np.zeros((len(T_GRID), P, Lg)), "tokens": np.zeros((len(T_GRID), P, Lg), dtype=np.int64),
           "hidden": {}, "norms": [], "label": label, "L_mid": L_mid}
    i_one = T_GRID.index(1.0)
    try:
        for it, T in enumerate(T_GRID):
            t0 = time.time()
            H_pts = []
            for b0 in range(0, P, batch):
                rows = list(range(b0, min(P, b0 + batch)))
                lens = [len(enc[r]) for r in rows]
                Lp = max(lens)
                ids = torch.full((len(rows), Lp), tok.pad_token_id if tok.pad_token_id is not None else 0, dtype=torch.long)
                attn = torch.zeros((len(rows), Lp), dtype=torch.long)
                for j, r in enumerate(rows):
                    ids[j, Lp - lens[j]:] = torch.tensor(enc[r])
                    attn[j, Lp - lens[j]:] = 1
                ids, attn = ids.to(emb_dev), attn.to(emb_dev)
                pos = (attn.cumsum(-1) - 1).clamp(min=0)
                past = None
                for s in range(Lg):
                    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    z = out.logits[:, -1, :].float()
                    if not torch.isfinite(z).all():
                        raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
                    h = cap["h"]
                    if s == 0 and it == 0 and b0 == 0:
                        first = [Lp - lens[j] for j in range(len(rows))]
                        n0 = torch.stack([h[j, first[j]].float().norm() for j in range(len(rows))]).mean().item()
                        nl = h[:, -1].float().norm(dim=-1).mean().item()
                        res["norms"].append(("prompt position 0 / dernière position du prompt", n0, nl))
                    if s >= 1:
                        for j, r in enumerate(rows):
                            if (r, s - 1) in sel:
                                H_pts.append(h[j, -1].float().cpu())
                    e1 = entropies_from_logits(z, Q_LIST)
                    eT = entropies_from_logits(z / T, Q_LIST)
                    for q in Q_LIST:
                        res["S_p1"][q][it, rows, s] = e1[q].cpu().numpy()
                        res["S_pT"][q][it, rows, s] = eT[q].cpu().numpy()
                    if it == i_one:
                        for jt, T2 in enumerate(T_GRID):
                            eF = entropies_from_logits(z / T2, Q_LIST)
                            for q in Q_LIST:
                                res["S_frozen"][q][jt, rows, s] = eF[q].cpu().numpy()
                    seeds = [SEED + 100003 * r + s for r in rows]
                    score = z / T + gumbel_noise(V, seeds, z.device)
                    score[:, list(forbid)] = -float("inf")
                    if V > n_tok:
                        score[:, n_tok:] = -float("inf")
                    nxt = torch.argmax(score, dim=-1)
                    res["surprisal"][it, rows, s] = (-torch.log_softmax(z.double(), -1).gather(1, nxt[:, None])[:, 0]).cpu().numpy()
                    res["tokens"][it, rows, s] = nxt.cpu().numpy()
                    ids = nxt[:, None].to(emb_dev)
                    attn = torch.cat([attn, torch.ones((len(rows), 1), dtype=attn.dtype, device=emb_dev)], dim=1)
                    pos = pos[:, -1:] + 1
                del past, out
                if b0 == 0 and it == 0:
                    est = (time.time() - t0) * math.ceil(P / batch) * len(T_GRID) / 60
                    print(f"[PROGRESSION] {label} : premier lot en {time.time() - t0:.1f} s ; estimation pour ce modèle ~{est:.0f} min", flush=True)
                    if est / 60 > MAX_HOURS:
                        raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
            res["hidden"][T] = torch.stack(H_pts) if H_pts else torch.zeros(0)
            if it == 0 and H_pts:
                res["norms"].append(("jetons générés (points TwoNN)", float(res["hidden"][T].norm(dim=-1).mean()), float(res["hidden"][T].norm(dim=-1).max())))
            dt = time.time() - t0
            left = dt * (len(T_GRID) - it - 1) / 60
            print(f"[MESURE] {label} T={T:<4} : {dt:6.1f} s (reste ~{left:.0f} min pour ce modèle), S_1,45(p1) moyen "
                  f"{res['S_p1'][Q_PRIMARY][it].mean():.4f}, S_1,45(pT) moyen {res['S_pT'][Q_PRIMARY][it].mean():.4f}")
    finally:
        hk.remove()
    return res

# %%
# ---------- Analyse ----------
def analyse(res, interpretable):
    rng = np.random.default_rng(SEED + 3)
    P = res["surprisal"].shape[1]
    out = {"label": res["label"]}
    ok = interpretable and POSITIVE_CONTROL_OK
    tag = None if ok else ("NON TESTABLE (témoin positif raté)" if interpretable and not POSITIVE_CONTROL_OK else None)

    per = {}
    for base in ("S_p1", "S_pT", "S_frozen"):
        for q in Q_LIST:
            S = res[base][q]
            per[(base, q, "G1")] = np.stack([g1_per_prompt(S[i]) for i in range(len(T_GRID))])
            per[(base, q, "G2")] = np.stack([g2_per_prompt(S[i]) for i in range(len(T_GRID))])
            per[(base, q, "meanS")] = S.mean(axis=2)
    msur = res["surprisal"].mean(axis=2)

    a1 = per[("S_p1", Q_PRIMARY, "G1")]
    full, stars, _ = bootstrap_argmax(lambda idx: curve_from_per_prompt(a1, idx), P, T_GRID, BOOT_N, rng)
    v1 = peak_verdict(full, stars, T_GRID, WINDOW, interpretable=ok)
    if tag:
        v1["verdict"] = tag
    out["V1"], out["G1_curve"] = v1, full.tolist()

    chi_full, chi_stars, _ = bootstrap_argmax(lambda idx: chi_curve(msur, T_GRID, GEN_LEN, idx), P, T_GRID, BOOT_N, rng)
    kc = int(np.argmax(chi_full))
    out["chi_curve"], out["chi_T_star"] = chi_full.tolist(), float(T_GRID[kc])
    out["chi_interior"] = bool(0 < kc < len(T_GRID) - 1)
    if out["chi_interior"]:
        w2 = (T_GRID[kc] - CHI_HALF_WIDTH, T_GRID[kc] + CHI_HALF_WIDTH)
        v2 = peak_verdict(full, stars, T_GRID, w2, interpretable=ok)
        v2["window"] = w2
        if tag:
            v2["verdict"] = tag
    else:
        v2 = {"verdict": "NON TESTABLE (χ sans maximum intérieur)" if interpretable else "NON INTERPRÉTABLE"}
    out["V2"] = v2

    desc = {}
    for base in ("S_p1", "S_pT", "S_frozen"):
        for q in Q_LIST:
            for g in ("G1", "G2"):
                c = per[(base, q, g)].mean(axis=1)
                desc[f"{base}|q={q}|{g}"] = {"T_star": float(T_GRID[int(np.argmax(c))]), "curve": c.tolist()}
            c3 = g3_curve(per[(base, q, "meanS")])
            desc[f"{base}|q={q}|G3"] = {"T_star": float(T_GRID[int(np.argmax(c3))]), "curve": c3.tolist()}
            desc[f"{base}|q={q}|niveau"] = {"curve": per[(base, q, "meanS")].mean(axis=1).tolist()}
    out["descriptif"] = desc

    tw = []
    for T in T_GRID:
        X = res["hidden"].get(T)
        if X is None or X.shape[0] < 20:
            tw.append((float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        Xd = X.to(DEVICE)
        d, dup = twonn(Xd)
        bs = []
        g = np.random.default_rng(SEED + 11)
        for _ in range(TWONN_BOOT):
            idx = torch.as_tensor(g.choice(X.shape[0], int(0.8 * X.shape[0]), replace=False), device=Xd.device)
            bs.append(twonn(Xd[idx])[0])   # sous-échantillonnage à 80 % sans remise (v2)
        tw.append((d, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), dup))
    out["twonn"] = tw
    out["per_prompt"] = {f"{k[0]}|q={k[1]}|{k[2]}": v for k, v in per.items()}
    out["mean_surprisal"] = msur
    return out


def print_analysis(a, res, tok):
    lab = a["label"]
    print(f"\n{'=' * 100}\n[RÉSULTATS] {lab}")
    for name, x, y in res["norms"]:
        print(f"[NORME] {name} : {x:.1f} / {y:.1f}")
    print(f"{'T':>5} | {'G1 p1 q=1,45':>13} | {'χ':>10} | {'TwoNN':>7} [{'IC95':^13}] | {'dupl.':>6} | "
          f"{'niveau S_1,45(pT)':>17} | {'G1 pT':>9} | {'G1 figé':>9}")
    for i, T in enumerate(T_GRID):
        d, lo, hi, dup = a["twonn"][i]
        print(f"{T:>5} | {a['G1_curve'][i]:>13.5f} | {a['chi_curve'][i]:>10.4f} | {d:>7.2f} [{lo:>5.2f} ; {hi:>5.2f}] | "
              f"{dup:>6.1%} | {a['descriptif'][f'S_pT|q={Q_PRIMARY}|niveau']['curve'][i]:>17.4f} | "
              f"{a['descriptif'][f'S_pT|q={Q_PRIMARY}|G1']['curve'][i]:>9.5f} | {a['descriptif'][f'S_frozen|q={Q_PRIMARY}|G1']['curve'][i]:>9.5f}")
    lv = [a["descriptif"][f"S_pT|q={q}|niveau"]["curve"] for q in Q_LIST]
    mono = all(np.all(np.diff(c) > 0) for c in lv)
    print(f"[P1] niveau de S_q(pT) strictement croissant en T pour tous les q : {'oui' if mono else 'NON'}")
    v1 = a["V1"]
    print(f"[VERDICT V1] {lab} : {v1['verdict']} — T*={v1['T_star']} (maximum {'intérieur' if v1['interior'] else 'AU BORD'}), "
          f"IC95 de T* [{v1['T_star_p2.5']} ; {v1['T_star_p97.5']}], fraction dans {WINDOW} : {v1['frac_in_window']:.3f}")
    v2 = a["V2"]
    if "frac_in_window" in v2:
        print(f"[VERDICT V2] {lab} : {v2['verdict']} — T*_χ={a['chi_T_star']}, fenêtre {v2['window']}, "
              f"fraction {v2['frac_in_window']:.3f}")
    else:
        print(f"[VERDICT V2] {lab} : {v2['verdict']} — T*_χ={a['chi_T_star']} (maximum au bord)")
    gen, frz = a["descriptif"][f"S_pT|q={Q_PRIMARY}|G1"], a["descriptif"][f"S_frozen|q={Q_PRIMARY}|G1"]
    diff = np.asarray(gen["curve"]) - np.asarray(frz["curve"])
    print(f"[TÉMOIN FIGÉ] base pT, q={Q_PRIMARY} : T* généré {gen['T_star']} contre figé {frz['T_star']} ; "
          f"généré − figé par T : {np.round(diff, 5).tolist()}")
    print("[DESCRIPTIF] T* (argmax) par signal, base et q — aucun verdict :")
    for base in ("S_p1", "S_pT", "S_frozen"):
        row = []
        for q in Q_LIST:
            row.append(f"q={q}: G1 {a['descriptif'][f'{base}|q={q}|G1']['T_star']}, "
                       f"G2 {a['descriptif'][f'{base}|q={q}|G2']['T_star']}, G3 {a['descriptif'][f'{base}|q={q}|G3']['T_star']}")
        print(f"   {base:<8} " + " ; ".join(row))
    for T in (0.2, 1.0, 2.0):
        it = T_GRID.index(T)
        for r in (0, len(PROMPTS) // 2 if len(PROMPTS) > 1 else 0):
            txt = tok.decode(res["tokens"][it, r, :60].tolist())
            print(f"[EXTRAIT] {lab} T={T} prompt {r} : {PROMPTS[r]!r} → {txt[:220]!r}")


# ---------- Volets v2 (W1 à W4) ----------
from scipy.stats import spearmanr


def centroid_T(curve, grid=None):
    g = np.asarray(T_GRID if grid is None else grid)
    w = np.asarray(curve) - np.min(curve)
    tr = getattr(np, "trapezoid", None) or np.trapz
    den = tr(w, g)
    return float(tr(w * g, g) / den) if den > 0 else float("nan")


def g1_masked(S, mask):
    """G1 par prompt sur les pas sélectionnés (différences seulement entre pas consécutifs tous deux sélectionnés)."""
    d = np.abs(np.diff(S, axis=-1))
    m = (mask[1:] & mask[:-1]).astype(float)
    return (d * m).sum(-1) / m.sum()


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def analyse_v2(res, interpretable):
    ok = interpretable and POSITIVE_CONTROL_OK
    rng = np.random.default_rng(SEED + 21)
    P, Lg = res["surprisal"].shape[1], res["surprisal"].shape[2]
    out = {}
    na = None if ok else ("NON INTERPRÉTABLE" if not interpretable else "NON TESTABLE (témoin positif raté)")
    idx_b = [rng.integers(0, P, P) for _ in range(BOOT_N)]

    # W1
    w1 = {}
    for q in Q_LIST:
        g = np.stack([g1_per_prompt(res["S_p1"][q][i]) for i in range(len(T_GRID))])
        stars = np.array([T_GRID[int(np.argmax(g[:, ix].mean(1)))] for ix in idx_b])
        w1[q] = {"T_star": float(T_GRID[int(np.argmax(g.mean(1)))]),
                 "frac": float(np.mean((stars >= WINDOW[0] - 1e-9) & (stars <= WINDOW[1] + 1e-9))),
                 "ci": ci(stars)}
    fr = [w1[q]["frac"] for q in Q_LIST]
    out["W1"] = {"par_q": w1, "verdict": na or ("SOUTENU" if min(fr) >= 0.95 else "RÉFUTÉ" if min(fr) <= 0.05 else "INDÉTERMINÉ")}

    # W2
    Ls = [L for L in L_SIZES if L <= Lg]
    mL = {L: res["surprisal"][:, :, :L].mean(2) for L in Ls}           # (nT, P)

    def chi_L(L, ix):
        a = mL[L][:, ix]
        return L * a.var(axis=1) / np.asarray(T_GRID) ** 2

    full = {L: chi_L(L, np.arange(P)) for L in Ls}
    k_star = int(np.argmax(full[Ls[-1]]))
    ks = [k_star, T_GRID.index(0.6), T_GRID.index(1.8)]
    logL = np.log(Ls)

    def slopes(ix):
        C = np.stack([chi_L(L, ix) for L in Ls])                       # (nL, nT)
        return [np.polyfit(logL, np.log(C[:, k] + 1e-12), 1)[0] for k in ks]

    b_full = slopes(np.arange(P))
    db = np.array([(lambda b: b[0] - max(b[1], b[2]))(slopes(ix)) for ix in idx_b])
    lo, hi = ci(db)
    out["W2"] = {"T_star_chi": float(T_GRID[k_star]), "b_peak": b_full[0], "b_0.6": b_full[1], "b_1.8": b_full[2],
                 "delta_b": float(b_full[0] - max(b_full[1], b_full[2])), "ci": (lo, hi),
                 "chi_par_L": {L: full[L].tolist() for L in Ls},
                 "verdict": na or ("SOUTENU" if lo > 0 else "RÉFUTÉ" if hi < 0.05 else "INDÉTERMINÉ")}

    # W3
    S = res["S_p1"][Q_PRIMARY]
    perm = np.argsort(np.random.default_rng(SEED + 31).random((P, Lg)), axis=1)
    g_real = np.stack([g1_per_prompt(S[i]) for i in range(len(T_GRID))])
    g_mel = np.stack([g1_per_prompt(np.take_along_axis(S[i], perm, axis=1)) for i in range(len(T_GRID))])
    same = np.array([abs(T_GRID[int(np.argmax(g_mel[:, ix].mean(1)))] - T_GRID[int(np.argmax(g_real[:, ix].mean(1)))]) <= 0.1 + 1e-9
                     for ix in idx_b])
    f3 = float(same.mean())
    out["W3"] = {"T_star_reel": float(T_GRID[int(np.argmax(g_real.mean(1)))]),
                 "T_star_melange": float(T_GRID[int(np.argmax(g_mel.mean(1)))]), "frac_meme_pic": f3,
                 "rapport_reel_sur_melange": (g_real.mean(1) / g_mel.mean(1)).tolist(),
                 "verdict": na or ("PIC EXPLIQUÉ PAR LA DISPERSION" if f3 >= 0.95 else "PIC DYNAMIQUE" if f3 <= 0.05 else "INDÉTERMINÉ")}

    # W4
    steps = np.arange(Lg)
    blk = (steps // BLOCK) % 2
    gA = np.stack([g1_masked(S[i], blk == 0) for i in range(len(T_GRID))])   # (nT, P)
    gB = np.stack([g1_masked(S[i], blk == 1) for i in range(len(T_GRID))])
    tA = np.array([centroid_T(gA[:, p]) for p in range(P)])
    tB = np.array([centroid_T(gB[:, p]) for p in range(P)])
    rho = spearmanr(tA, tB).statistic
    prm = np.random.default_rng(SEED + 41)
    null = np.array([spearmanr(tA, prm.permutation(tB)).statistic for _ in range(5000)])
    pval = float((np.sum(null >= rho) + 1) / (len(null) + 1))
    rb = np.array([spearmanr(tA[ix], tB[ix]).statistic for ix in idx_b])
    rlo, rhi = ci(rb)
    out["W4a"] = {"rho": float(rho), "p_perm": pval, "ci": (rlo, rhi),
                  "verdict": na or ("SOUTENU" if rho > 0 and pval < 0.01 else "RÉFUTÉ" if (rlo >= -0.1 and rhi <= 0.1) else "INDÉTERMINÉ")}

    q4 = max(1, Lg // 4)
    early, late = steps < q4, steps >= Lg - q4
    tE = np.array([centroid_T(np.stack([g1_masked(S[i], early) for i in range(len(T_GRID))])[:, p]) for p in range(P)])
    tL = np.array([centroid_T(np.stack([g1_masked(S[i], late) for i in range(len(T_GRID))])[:, p]) for p in range(P)])
    dlt = tL - tE
    dm = np.array([np.nanmean(dlt[ix]) for ix in idx_b])
    dlo, dhi = ci(dm)
    out["W4b"] = {"diff_moyenne": float(np.nanmean(dlt)), "ci": (dlo, dhi),
                  "Tc_debut": float(np.nanmean(tE)), "Tc_fin": float(np.nanmean(tL)),
                  "verdict": na or ("SOUTENU" if (dlo > 0 or dhi < 0) else "RÉFUTÉ" if (dlo >= -0.03 and dhi <= 0.03) else "INDÉTERMINÉ")}

    i1 = T_GRID.index(1.0)
    inst = S[i1][:, :min(16, Lg)].mean(1)
    tall = np.array([centroid_T(g_real[:, p]) for p in range(P)])
    r4 = spearmanr(inst, tall).statistic
    rcb = np.array([spearmanr(inst[ix], tall[ix]).statistic for ix in idx_b])
    clo, chi_ = ci(rcb)
    out["W4c"] = {"rho": float(r4), "ci": (clo, chi_),
                  "verdict": na or ("SOUTENU" if chi_ < 0 else "CONTREDIT" if clo > 0 else "INDÉTERMINÉ")}
    out["Tc_par_prompt"] = tall.tolist()
    return out


def print_v2(lab, v):
    print(f"\n[W1] {lab} : {v['W1']['verdict']} — fraction dans {WINDOW} par q : "
          + " ; ".join(f"q={q}: T*={d['T_star']} IC95 [{d['ci'][0]} ; {d['ci'][1]}] frac {d['frac']:.3f}" for q, d in v["W1"]["par_q"].items()))
    w2 = v["W2"]
    print(f"[W2] {lab} : {w2['verdict']} — T*_χ(L max)={w2['T_star_chi']} ; b(T*)={w2['b_peak']:.3f}, b(0,6)={w2['b_0.6']:.3f}, "
          f"b(1,8)={w2['b_1.8']:.3f} ; Δb={w2['delta_b']:.3f} IC95 [{w2['ci'][0]:.3f} ; {w2['ci'][1]:.3f}]")
    for L, c in w2["chi_par_L"].items():
        print(f"     χ_L={L:<4} : " + " ".join(f"{x:8.2f}" for x in c))
    w3 = v["W3"]
    print(f"[W3] {lab} : {w3['verdict']} — T* réel {w3['T_star_reel']} contre mélangé {w3['T_star_melange']} ; "
          f"fraction même pic (±0,1) {w3['frac_meme_pic']:.3f} ; réel/mélangé par T : {np.round(w3['rapport_reel_sur_melange'], 3).tolist()}")
    a, b, c = v["W4a"], v["W4b"], v["W4c"]
    print(f"[W4a] {lab} : {a['verdict']} — ρ(T_c blocs A, B) = {a['rho']:.3f}, p perm = {a['p_perm']:.4f}, IC95 [{a['ci'][0]:.3f} ; {a['ci'][1]:.3f}]")
    print(f"[W4b] {lab} : {b['verdict']} — T_c début {b['Tc_debut']:.3f}, fin {b['Tc_fin']:.3f}, différence {b['diff_moyenne']:+.3f} "
          f"IC95 [{b['ci'][0]:+.3f} ; {b['ci'][1]:+.3f}]")
    print(f"[W4c] {lab} : {c['verdict']} — ρ(instabilité initiale, T_c) = {c['rho']:.3f}, IC95 [{c['ci'][0]:.3f} ; {c['ci'][1]:.3f}]")
    tc = np.array(v["Tc_par_prompt"])
    print(f"[W4] {lab} : T_c par prompt, médiane {np.nanmedian(tc):.3f}, quantiles 10–90 % [{np.nanpercentile(tc, 10):.3f} ; {np.nanpercentile(tc, 90):.3f}]")

# %%
# ---------- Boucle principale ----------
interpretable = (not SMOKE_TEST) and len(PROMPTS) >= N_MIN
ALL, RAW = {}, {}
for label, ids in DEFAULT_MODELS:
    t_start = time.time()
    if (t_start - T_RUN0) / 3600 > MAX_HOURS:
        print(f"[BUDGET] {label} sauté : budget de {MAX_HOURS} h dépassé")
        ALL[label] = {"label": label, "V1": {"verdict": "NON TESTABLE (budget de temps)"}, "V2": {"verdict": "NON TESTABLE (budget de temps)"}}
        continue
    model = tok = None
    used = None
    err = None
    for mid in ids:
        print(f"[CHARGEMENT] {label} : téléchargement / chargement de {mid}…", flush=True)
        try:
            model, tok, mode = load_model(mid)
            used = mid
            print(f"[CHARGEMENT] {label} : {mid} en {mode} ({sum(p.numel() for p in model.parameters()) / 1e9:.2f} G paramètres)")
            break
        except Exception as e:
            err = f"{mid} : {type(e).__name__}: {str(e)[:300]}"
            print(f"[CHARGEMENT] échec {err}")
            model = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    if model is None:
        ALL[label] = {"label": label, "V1": {"verdict": "NON TESTABLE (modèle non chargé)"},
                      "V2": {"verdict": "NON TESTABLE (modèle non chargé)"}, "erreur": err}
        continue
    try:
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        res = run_model(model, tok, label)
        a = analyse(res, interpretable)
        a["model_id"], a["mode"], a["minutes"] = used, mode, (time.time() - t_start) / 60
        print_analysis(a, res, tok)
        a['v2'] = analyse_v2(res, interpretable)
        print_v2(label, a['v2'])
        ALL[label], RAW[label] = a, res
        np.savez_compressed(os.path.join(OUT, f"{label}_mesures.npz"),
                            T_grid=np.asarray(T_GRID), tokens=res["tokens"], surprisal=res["surprisal"],
                            **{f"S_p1_q{q}": res["S_p1"][q].astype(np.float32) for q in Q_LIST},
                            **{f"S_pT_q{q}": res["S_pT"][q].astype(np.float32) for q in Q_LIST},
                            **{f"S_frozen_q{q}": res["S_frozen"][q].astype(np.float32) for q in Q_LIST})
        texts = {str(T): [tok.decode(res["tokens"][it, r].tolist()) for r in range(len(PROMPTS))]
                 for it, T in enumerate(T_GRID)}
        with open(os.path.join(OUT, f"{label}_textes.json"), "w") as f:
            json.dump({"prompts": PROMPTS, "textes": texts}, f, ensure_ascii=False)
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min")
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}")
        ALL[label] = {"label": label, "V1": {"verdict": f"NON TESTABLE ({type(e).__name__})"},
                      "V2": {"verdict": f"NON TESTABLE ({type(e).__name__})"}, "erreur": str(e)[:500]}
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if HF_CACHE.startswith("/tmp") and used and used != "smoke-qwen3-random":
        subprocess.run(["rm", "-rf", HF_CACHE], check=False)

# %%
# ---------- Tableau final, CSV et graphiques ----------
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"\n{'=' * 100}\n[SYNTHÈSE] (SMOKE_TEST={SMOKE_TEST}, témoin positif {'réussi' if POSITIVE_CONTROL_OK else 'RATÉ'})")
for label, a in ALL.items():
    v1, v2 = a["V1"], a["V2"]
    extra = f"T*={v1.get('T_star')} fraction={v1.get('frac_in_window', float('nan')):.3f}" if "T_star" in v1 else a.get("erreur", "")
    print(f"[SYNTHÈSE] {label:<18} V1 : {v1['verdict']:<36} V2 : {v2['verdict']:<40} {extra}")
    if "v2" in a:
        w = a["v2"]
        print(f"[SYNTHÈSE] {label:<18} W1 : {w['W1']['verdict']} | W2 : {w['W2']['verdict']} | W3 : {w['W3']['verdict']} | "
              f"W4a : {w['W4a']['verdict']} | W4b : {w['W4b']['verdict']} | W4c : {w['W4c']['verdict']}")

with open(os.path.join(OUT, "par_prompt.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["modele", "T", "prompt", "signal", "valeur"])
    for label, a in ALL.items():
        if "per_prompt" not in a:
            continue
        for key, arr in a["per_prompt"].items():
            for it, T in enumerate(T_GRID):
                for r in range(arr.shape[1]):
                    w.writerow([label, T, r, key, f"{arr[it, r]:.8g}"])
        for it, T in enumerate(T_GRID):
            for r in range(a["mean_surprisal"].shape[1]):
                w.writerow([label, T, r, "surprisal_moyenne", f"{a['mean_surprisal'][it, r]:.8g}"])

summary = {label: {k: v for k, v in a.items() if k not in ("per_prompt", "mean_surprisal")} for label, a in ALL.items()}
with open(os.path.join(OUT, "resume.json"), "w") as f:
    json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "T_grid": T_GRID,
               "window": WINDOW, "q_primary": Q_PRIMARY, "modeles": summary}, f, ensure_ascii=False, indent=1, default=str)

ok_models = [l for l, a in ALL.items() if "G1_curve" in a]
if ok_models:
    fig, axes = plt.subplots(len(ok_models), 3, figsize=(15, 4 * len(ok_models)), squeeze=False)
    for row, label in enumerate(ok_models):
        a = ALL[label]
        ax = axes[row]
        ax[0].plot(T_GRID, a["G1_curve"], "o-", label="G1 p1 (V1)")
        ax[0].axvspan(*WINDOW, color="0.9")
        ax[0].set_title(f"{label} — G1, q={Q_PRIMARY}")
        ax[1].plot(T_GRID, a["chi_curve"], "o-", color="C1")
        ax[1].set_title("χ (susceptibilité, séquence)")
        d = [t[0] for t in a["twonn"]]
        ax[2].plot(T_GRID, d, "o-", color="C2")
        ax[2].set_title("TwoNN (couche médiane)")
        for x in ax:
            x.set_xlabel("température T")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "courbes.png"), dpi=110)
    print(f"[SORTIE] graphique : {os.path.join(OUT, 'courbes.png')}")
else:
    print("[SORTIE] aucun modèle mesuré : pas de graphique")
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

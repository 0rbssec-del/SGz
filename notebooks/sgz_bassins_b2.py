# %% [markdown]
# # SGZ B2 — Les trajectoires d'un même prompt forment-elles des bassins distincts ou un continuum ? (branche 2)
#
# **Question.** L'idée d'une carte de navigation (atlas, orbites, bifurcations) n'a un fondement que si, pour un même prompt,
# les suites tirées se rangent en quelques régimes séparés. Si elles forment un continuum, une carte ne montrera que du bruit.
# B2 répond à cette question préalable avant de construire l'outil.
#
# **Antériorité (lue le 29 oct.).** Bigelow et al. (ICLR 2025, arXiv 2412.07961) : en rééchantillonnant à chaque jeton d'un
# chemin de base (GPT-3.5, 7 tâches, 30 exemples par tâche), la distribution des *réponses finales* change brusquement à
# certains « jetons de bifurcation », parfois une simple ponctuation. Zur et al. (ICML 2025, arXiv 2511.04527) : chez
# Llama-3.2-3B-Instruct, les états cachés prédisent cette distribution des réponses, et l'orientation par activations marche
# surtout avant l'engagement (4 à 10 exemples). Dans les deux cas, l'issue est une réponse discrète (A, B, C, D ou un nombre) :
# les « bassins » y sont donnés par la tâche. La question pour un texte libre, sans réponse finale, reste ouverte (non trouvée
# dans cette recherche courte).
#
# **Dispositif.** Les 64 prompts écrits à la main de A1c (32 en français, 32 en anglais), 192 suites libres de 32 jetons par
# prompt, T = 0,6 (principal) et T = 1,0 (descriptif). Deux modèles : Qwen3-1.7B-Base et SmolLM3-3B-Base.
# Deux représentations d'une trajectoire :
# - **R_h** (principale) : moyenne, sur les 32 jetons de la suite, de l'état caché après les ⌊2L/3⌋ premières couches ;
# - **R_S** (la variable de l'atlas) : le profil d'entropie de Tsallis (q = 1,45) aux 32 positions.
#
# **Test de bimodalité par prompt (moitiés indépendantes).** Les 192 suites sont coupées en deux moitiés A et B (96 chacune).
# Sur A : standardisation, 10 composantes principales, k-moyennes à 2 groupes, direction qui joint les deux centres. Sur B
# (jamais vue par l'ajustement) : projection sur cette direction, puis test du dip de Hartigan (nul = loi unimodale ; robuste
# à l'asymétrie). Correction de Benjamini-Hochberg à 5 % sur les 64 prompts. π = part des prompts multimodaux.
#
# ## Critères (fixés avant toute exécution)
#
# - **Calibration** (substitut gaussien) : pour chaque prompt, 192 points tirés d'une gaussienne de même moyenne et même
#   covariance que les vraies suites, passés dans le même test. Part de rejets > 10 % → verdict NON INTERPRÉTABLE.
# - **Puissance** (mélanges réels) : 32 paires de prompts, 96 suites de chaque, passées dans le même test. Part détectée
#   < 80 % → un verdict CONTINUUM devient NON INTERPRÉTABLE (test sans puissance).
# - **B2-h (principal), R_h à T = 0,6** : BASSINS si π ≥ 0,5 ; CONTINUUM si π ≤ 0,1 ; sinon MIXTE (des bassins pour certains
#   prompts seulement).
# - **B2-S, R_S à T = 0,6** : même règle. Il dit si l'entropie seule suffit à voir les bassins.
# - **Synthèse, thèse « les trajectoires d'un même prompt se rangent en bassins distincts »** (sur B2-h) : SOUTENUE si BASSINS
#   pour les deux modèles ; RÉFUTÉE si CONTINUUM pour les deux ; DÉPEND DU MODÈLE si BASSINS pour l'un et CONTINUUM pour
#   l'autre ; sinon PARTIELLE.
# - **Descriptifs** : T = 1,0 avec les mêmes règles ; couche finale ; part de suites distinctes ; pour les prompts
#   multimodaux en R_h, position de la bifurcation k* (premier k où le préfixe de k jetons prédit le bassin à ≥ 90 %, et à au
#   moins 25 points au-dessus du hasard) et entropie à k* rapportée à celle des autres positions (prépare la branche 3) ;
#   exemples de suites de chaque bassin.
# - Témoins synthétiques du test (gaussienne anisotrope, loi asymétrique unimodale, mélange de deux groupes) et des règles.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# B2-h : BASSINS 25 %, MIXTE 45 %, CONTINUUM 30 %. B2-S : CONTINUUM 60 %. Synthèse SOUTENUE 20 %, RÉFUTÉE 25 %. Pour les
# prompts multimodaux : bifurcation dans les 4 premiers jetons 60 % ; entropie à la bifurcation plus de 1,5 fois celle des
# autres positions 70 %.
#
# ## Limites connues
#
# - « Bassin » dépend de la représentation et de l'horizon : un continuum à 32 jetons peut se séparer plus tard, et
#   l'inverse.
# - Le test ne voit qu'une bimodalité le long d'une direction (conservateur) ; la moyenne des états cachés peut lisser des
#   différences réelles de contenu.
# - Des suites identiques (fréquentes si le modèle est très sûr) forment des atomes ; elles sont gardées et leur part est
#   rapportée.
# - Deux modèles de base, prompts courts, anglais et français seulement.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis ; le paquet `diptest` est installé par
# pip). Durée estimée : ~15 min pour Qwen3, ~30 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN]`,
# `[CALIBRATION]`, `[PUISSANCE]`, `[B2-h]`, `[B2-S]`, `[SYNTHÈSE]`.

# %%
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
try:
    import diptest
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diptest"], check=False)
    try:
        import diptest
    except ImportError:
        raise RuntimeError("paquet diptest indisponible : active Internet (Settings → Internet on) puis relance")
from sklearn.cluster import KMeans

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_b2" if os.path.isdir("/kaggle/working") else "sgz_b2_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

TEMPS = [0.6, 1.0]                               # T = 0,6 principal, T = 1,0 descriptif
Q = 1.45
N_PROMPTS, N_ECH, NSUITE, NPC = 64, 192, 32, 10
ALPHA = 0.05
SEUIL_BASSINS, SEUIL_CONTINUUM, SEUIL_NUL, SEUIL_PUISSANCE = 0.5, 0.1, 0.1, 0.8
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261029
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, N_ECH, NSUITE, NPC = int(os.environ.get("SMOKE_N", "8")), 48, 12, 5
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={TEMPS} prompts={N_PROMPTS} suites par prompt={N_ECH} longueur={NSUITE} composantes={NPC} ; "
      f"modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}, diptest {getattr(diptest, '__version__', '?')}")

# %%
# ---------- Prompts (les 64 écrits à la main de A1c) ----------
PROMPTS_MAIN = [
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
if SMOKE_TEST:                                   # moitié français, moitié anglais
    PROMPTS = PROMPTS_MAIN[:N_PROMPTS // 2] + PROMPTS_MAIN[32:32 + N_PROMPTS - N_PROMPTS // 2]
else:
    PROMPTS = PROMPTS_MAIN[:N_PROMPTS]
PAIRES = [(2 * i, 2 * i + 1) for i in range(len(PROMPTS) // 2)]   # mélanges de deux prompts (puissance)
print(f"[PROMPTS] {len(PROMPTS)} prompts, {len(PAIRES)} paires pour la puissance")

# %%
# ---------- Mesures, test de bimodalité, règles et témoins ----------
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


def gumbel_noise(V, seeds, device):
    """Bruit de Gumbel par ligne, graine explicite par ligne."""
    g = torch.Generator(device=device)
    rows = []
    for s in seeds:
        g.manual_seed(int(s))
        rows.append(torch.rand(V, generator=g, device=device, dtype=torch.float32))
    u = torch.stack(rows).clamp_(1e-10, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - h), float(c + h))


def standardiser(X, ref):
    mu, sd = ref.mean(0), ref.std(0)
    sd = sd + 1e-6 * sd.max() + 1e-12
    return (X - mu) / sd


def acp(Z, k):
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return Vt[:min(k, Vt.shape[0])].T


def test_bimodal(X, seed):
    """Moitiés indépendantes : direction des 2-moyennes ajustée sur A, test du dip sur la projection de B."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    A, B = X[perm[:len(X) // 2]], X[perm[len(X) // 2:]]
    Az, Bz = standardiser(A, A), standardiser(B, A)
    V = acp(Az, NPC)
    PA, PB = Az @ V, Bz @ V
    if np.ptp(PA, axis=0).max() < 1e-9:
        return 0.0, 1.0
    km = KMeans(2, n_init=10, random_state=int(seed % 2 ** 31)).fit(PA)
    u = km.cluster_centers_[0] - km.cluster_centers_[1]
    u = u / (np.linalg.norm(u) + 1e-12)
    proj = PB @ u
    if np.ptp(proj) < 1e-9:
        return 0.0, 1.0
    d, p = diptest.diptest(proj.astype(np.float64))
    return float(d), float(p)


def substitut_gaussien(X, rng):
    """Même moyenne et même covariance empirique que X, loi gaussienne (unimodale)."""
    Xc = X - X.mean(0)
    return X.mean(0) + rng.standard_normal((len(X), len(X))) @ Xc / np.sqrt(len(X) - 1)


def bh(p, alpha=ALPHA):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    k = int(np.nonzero(ok)[0].max()) + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[o[:k]] = True
    return rej


def regle_b2(pi, nul, puis, interpretable=True):
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if nul > SEUIL_NUL:
        return "NON INTERPRÉTABLE (test mal calibré sur le substitut gaussien)"
    if pi >= SEUIL_BASSINS:
        return "BASSINS"
    if pi <= SEUIL_CONTINUUM:
        return "CONTINUUM" if puis >= SEUIL_PUISSANCE else "NON INTERPRÉTABLE (test sans puissance)"
    return "MIXTE"


def synthese(v, interpretable=True):
    """v : dict modèle → verdict B2-h."""
    ms = list(v)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms or any(x.startswith("NON") for x in v.values()):
        return "INDÉTERMINÉE"
    if all(v[m] == "BASSINS" for m in ms):
        return "SOUTENUE"
    if all(v[m] == "CONTINUUM" for m in ms):
        return "RÉFUTÉE"
    if set(v.values()) == {"BASSINS", "CONTINUUM"}:
        return "DÉPEND DU MODÈLE"
    return "PARTIELLE"


# Témoins synthétiques du test (192 points en dimension 64, 20 répétitions chacun)
rng_t = np.random.default_rng(SEED + 1)
_npc, NPC = NPC, 10
echelle = np.exp(rng_t.normal(0, 1, 64))


def taux(gen, nrep=20):
    return float(np.mean([test_bimodal(gen(), SEED + 10 + r)[1] < ALPHA for r in range(nrep)]))


def melange(delta):
    """Deux groupes séparés de delta écarts-types le long d'une direction répartie sur toutes les dimensions (comme une différence
    de contenu), puis échelles inégales par dimension."""
    u = rng_t.normal(0, 1, 64)
    u /= np.linalg.norm(u)
    X = rng_t.normal(0, 1, (192, 64))
    X[:96] += delta / 2 * u
    X[96:] -= delta / 2 * u
    return X * echelle


r_gauss = taux(lambda: rng_t.normal(0, 1, (192, 64)) * echelle)
def asymetrique():
    """Loi unimodale très asymétrique (log-normale) le long d'une direction répartie, dominante en variance."""
    u = rng_t.normal(0, 1, 64)
    u /= np.linalg.norm(u)
    return (rng_t.normal(0, 1, (192, 64)) + 3 * (rng_t.lognormal(0, 1, (192, 1)) - np.exp(0.5)) * u) * echelle


r_asym = taux(asymetrique)
r_m4, r_m6 = taux(lambda: melange(4.0)), taux(lambda: melange(6.0))
NPC = _npc
checks = [("gaussienne anisotrope : rejets ≤ 15 %", r_gauss <= 0.15),
          ("loi asymétrique unimodale : rejets ≤ 15 %", r_asym <= 0.15),
          ("mélange à 6 écarts-types : rejets ≥ 80 %", r_m6 >= 0.8),
          ("règle BASSINS", regle_b2(0.6, 0.0, 0.9) == "BASSINS"),
          ("règle CONTINUUM", regle_b2(0.05, 0.0, 0.9) == "CONTINUUM"),
          ("règle MIXTE", regle_b2(0.3, 0.0, 0.9) == "MIXTE"),
          ("règle sans puissance", regle_b2(0.05, 0.0, 0.5) == "NON INTERPRÉTABLE (test sans puissance)"),
          ("règle mal calibrée", regle_b2(0.6, 0.2, 0.9) == "NON INTERPRÉTABLE (test mal calibré sur le substitut gaussien)"),
          ("synthèse soutenue", synthese({"a": "BASSINS", "b": "BASSINS"}) == "SOUTENUE"),
          ("synthèse réfutée", synthese({"a": "CONTINUUM", "b": "CONTINUUM"}) == "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "BASSINS", "b": "CONTINUUM"}) == "DÉPEND DU MODÈLE"),
          ("synthèse partielle", synthese({"a": "MIXTE", "b": "BASSINS"}) == "PARTIELLE"),
          ("BH", bh([0.001, 0.02, 0.5, 0.04]).tolist() == [True, True, False, False])]
print(f"[TÉMOIN] taux de rejet du test : gaussienne {r_gauss:.0%}, asymétrique {r_asym:.0%}, mélange à 4 écarts-types {r_m4:.0%}, "
      f"à 6 écarts-types {r_m6:.0%}")
for nom, okc in checks:
    print(f"[TÉMOIN] {nom:<44} : {'OK' if okc else 'RATÉ'}")
POSITIVE_CONTROL_OK = all(c for _, c in checks)
print(f"[TÉMOIN] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON INTERPRÉTABLE'}")

# %%
# ---------- Chargement ----------
class SmokeTokenizer:
    """Tokenizer jouet par caractères, aller-retour exact (SMOKE_TEST)."""
    ALPHA = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-«»\n"
             "àâäçéèêëîïôöùûüÀÇÉÈÊ")

    def __init__(self, V):
        self.V, self.pad_token_id, self.eos_token_id = V, 0, 1
        self.all_special_ids = [0, 1]

    def __len__(self):                          # jetons valides : l'alphabet (le reste du vocabulaire est masqué)
        return 2 + len(self.ALPHA)

    def encode(self, s, **kw):
        return [2 + self.ALPHA.index(c) if c in self.ALPHA else 2 + self.ALPHA.index("?") for c in s]

    def decode(self, ids, **kw):
        return "".join(self.ALPHA[i - 2] if 2 <= i < 2 + len(self.ALPHA) else "?" for i in ids)


def transformers_major():
    return int(transformers.__version__.split(".")[0])


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == "smoke-qwen3-random":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
        torch.manual_seed(SEED)
        return Qwen3ForCausalLM(cfg).eval().to(DEVICE), SmokeTokenizer(512), "fp32 aléatoire (smoke)"
    if model_id == "smoke-qwen3-texte":
        # Qwen3 minuscule entraîné par caractères : chaque prompt est suivi, au hasard, de l'une de deux suites très différentes.
        # Les bassins existent donc par construction (témoin positif de tout le pipeline, harnais local seulement).
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512)
        torch.manual_seed(SEED)
        m, tk = Qwen3ForCausalLM(cfg).to(DEVICE), SmokeTokenizer(512)
        rng = np.random.default_rng(SEED + 99)
        deux = [" et puis le soleil brille sur la ville, les enfants jouent dehors.",
                " mais la pluie tombe sans fin sur les toits gris ce soir-là."]
        corpus = np.array(tk.encode(" ".join(PROMPTS_MAIN[i % 64] + deux[rng.integers(2)] for i in range(4000))))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for _ in range(int(os.environ.get("SMOKE_PAS", "800"))):
            d = rng.integers(0, len(corpus) - 129, 32)
            x = torch.tensor(np.stack([corpus[i:i + 128] for i in d]), device=DEVICE)
            loss = m(input_ids=x, labels=x).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m.eval(), tk, f"fp32 entraîné sur prompts + deux suites (smoke, perte finale {loss.item():.2f})"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    else:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": 0}, "fp16 1 GPU"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda":
        assert next(model.parameters()).dtype == torch.float16, "type de poids inattendu"
    return model, tok, mode

# %%
# ---------- Génération (comme A1c–B1) ----------
class Gen:
    def __init__(self, model, tok, label):
        self.model, self.tok, self.label = model, tok, label
        self.dev = model.get_input_embeddings().weight.device
        self.V = model.get_output_embeddings().weight.shape[0]
        forbid = set(tok.all_special_ids or []) | {tok.eos_token_id}
        ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        forbid |= set(ge if isinstance(ge, (list, tuple)) else ([ge] if ge is not None else []))
        forbid.discard(None)
        self.forbid, self.n_tok = sorted(forbid), len(tok)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        npar = sum(p.numel() for p in model.parameters())
        self.budget = 48 if npar < 3e9 else 32

    def left_pad(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.dev), attn.to(self.dev), L

    @torch.no_grad()
    def sample(self, ctxs, prompt_of, T, nsteps, step0, base=SEED):
        """Graine du pas s de la ligne j : base + 100003·prompt_of[j] + step0 + s."""
        n = len(ctxs)
        S, tk = np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int64)
        ids, attn, _ = self.left_pad(ctxs)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        past = None
        for s in range(nsteps):
            out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
            e1 = entropies_from_logits(z, [Q])[Q].cpu().numpy()
            seeds = [base + 100003 * int(prompt_of[j]) + step0 + s for j in range(n)]
            score = z / T + gumbel_noise(self.V, seeds, z.device)
            score[:, self.forbid] = -float("inf")
            if self.V > self.n_tok:
                score[:, self.n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            S[:, s], tk[:, s] = e1, nxt.cpu().numpy()
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = pos[:, -1:] + 1
        del past, out
        return S, tk

# %%
# ---------- Suites, états cachés, bifurcations ----------
@torch.no_grad()
def tirer_suites(g, enc, T, ti, annonce=True):
    """N_ECH suites libres de NSUITE jetons par prompt ; graine propre à chaque suite (prompt p, tirage j, température ti)."""
    P = len(enc)
    S = np.zeros((P, N_ECH, NSUITE), np.float32)
    TK = np.zeros((P, N_ECH, NSUITE), np.int64)
    jobs = [(p, j) for p in range(P) for j in range(N_ECH)]
    nb = (len(jobs) + g.budget - 1) // g.budget
    for b, b0 in enumerate(range(0, len(jobs), g.budget)):
        t0 = time.time()
        lot = jobs[b0:b0 + g.budget]
        s, t = g.sample([enc[p] for p, _ in lot], [p * 256 + j for p, j in lot], T, NSUITE, 0, base=SEED + 17 * (ti + 1))
        for (p, j), sr, tr in zip(lot, s, t):
            S[p, j], TK[p, j] = sr, tr
        if b == 0 and ti == 0 and annonce:
            est = (time.time() - t0) * nb * len(TEMPS) / 60
            print(f"[PROGRESSION] {g.label} : premier lot en {time.time() - t0:.1f} s ; estimation ~{est:.0f} min (génération)",
                  flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
    return S, TK


@torch.no_grad()
def etats_moyens(g, enc, TK, couches):
    """Moyenne, sur les NSUITE jetons de chaque suite, de l'état caché après les couches demandées (indices de hidden_states)."""
    P = len(enc)
    out = {c: None for c in couches}
    jobs = [(p, j) for p in range(P) for j in range(N_ECH)]
    for b0 in range(0, len(jobs), g.budget):
        lot = jobs[b0:b0 + g.budget]
        ids, attn, L = g.left_pad([enc[p] + TK[p, j].tolist() for p, j in lot])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        hs = g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False,
                           output_hidden_states=True).hidden_states
        for c in couches:
            m = hs[c][:, L - NSUITE:, :].float().mean(1).cpu().numpy()
            if out[c] is None:
                out[c] = np.zeros((P, N_ECH, m.shape[1]), np.float32)
            for (p, j), row in zip(lot, m):
                out[c][p, j] = row
        del hs
    return out


def purete(cles, lab):
    grp = {}
    for c, l in zip(cles, lab):
        grp.setdefault(c, []).append(l)
    return sum(max(np.bincount(v)) for v in grp.values()) / len(lab)


def bifurcation(TK, lab, S, rng):
    """Premier k où le préfixe de k jetons prédit le bassin à ≥ 90 %, à ≥ 25 points au-dessus du hasard (étiquettes permutées)."""
    acc, hasard = [], []
    for k in range(1, NSUITE + 1):
        cles = [tuple(r[:k]) for r in TK]
        acc.append(purete(cles, lab))
        hasard.append(float(np.mean([purete(cles, rng.permutation(lab)) for _ in range(20)])))
    acc, hasard = np.array(acc), np.array(hasard)
    ok = np.nonzero((acc >= 0.9) & (acc - hasard >= 0.25))[0]
    k = int(ok[0]) + 1 if ok.size else None
    prof = S.mean(0)
    ratio = float(prof[k - 1] / np.median(np.delete(prof, k - 1))) if k else float("nan")
    return k, acc, hasard, ratio


def etiquettes(X, seed):
    Z = standardiser(X, X)
    return KMeans(2, n_init=10, random_state=seed).fit_predict(Z @ acp(Z, NPC))

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_b2.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK,
                   "temoins": {"gaussienne": r_gauss, "asymetrique": r_asym, "melange4": r_m4, "melange6": r_m6}, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def analyser(R, nom, label, Tv, seed0):
    """R : (P, N_ECH, d). Test par prompt, substitut gaussien, mélanges de paires, verdict."""
    P = R.shape[0]
    rng = np.random.default_rng(seed0)
    vrai = [test_bimodal(R[p].astype(np.float64), seed0 + p) for p in range(P)]
    nul = [test_bimodal(substitut_gaussien(R[p].astype(np.float64), rng), seed0 + 1000 + p) for p in range(P)]
    mix = []
    for a, b in PAIRES:
        ia, ib = rng.permutation(N_ECH)[:N_ECH // 2], rng.permutation(N_ECH)[:N_ECH // 2]
        mix.append(test_bimodal(np.concatenate([R[a][ia], R[b][ib]]).astype(np.float64), seed0 + 2000 + a))
    rej = bh([p for _, p in vrai])
    pi, tn, pu = float(rej.mean()), float(bh([p for _, p in nul]).mean()), float(bh([p for _, p in mix]).mean())
    v = regle_b2(pi, tn, pu, ok)
    ic = wilson(int(rej.sum()), P)
    print(f"[{nom}] {label} T = {Tv} : prompts multimodaux {int(rej.sum())}/{P} (π = {pi:.2f}, IC95 [{ic[0]:.2f} ; {ic[1]:.2f}]) ; "
          f"dip médian {np.median([d for d, _ in vrai]):.4f} ; substitut gaussien {tn:.0%} ; mélanges de deux prompts {pu:.0%} → {v}")
    return {"pi": pi, "pi_ci": ic, "n_multi": int(rej.sum()), "nul": tn, "puissance": pu, "verdict": v,
            "dip": [d for d, _ in vrai], "p": [p for _, p in vrai], "multimodal": rej.tolist()}


for label, ids_m in MODELS:
    t_start = time.time()
    model = tok = g = None
    try:
        for mid in ids_m:
            print(f"\n[CHARGEMENT] {label} : téléchargement / chargement de {mid}…", flush=True)
            try:
                model, tok, mode = load_model(mid)
                print(f"[CHARGEMENT] {label} : {mid} en {mode}")
                break
            except Exception as e:
                print(f"[CHARGEMENT] échec {mid} : {type(e).__name__}: {str(e)[:300]}")
                model = None
        if model is None:
            raise RuntimeError("modèle non chargé")
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        g = Gen(model, tok, label)
        enc = [tok.encode(p) for p in PROMPTS]
        nL = model.config.num_hidden_layers
        couches = [(2 * nL) // 3, nL]                # après ⌊2L/3⌋ couches (principal) ; sortie finale normalisée (descriptif)
        a = {"couches": couches}
        for ti, Tv in enumerate(TEMPS):
            t1 = time.time()
            S, TK = tirer_suites(g, enc, Tv, ti)
            H = etats_moyens(g, enc, TK, couches)
            if ti == 0:
                b_ = g.budget
                g.budget = max(1, b_ // 2 - 5)       # autres lots, à cheval sur deux prompts (remplissage à gauche différent)
                S2, TK2 = tirer_suites(g, enc[:2], Tv, ti, annonce=False)
                g.budget = b_
                same = float(np.mean(TK2 == TK[:2]))
                print(f"[VÉRIF] {label} reprise des suites des 2 premiers prompts dans des lots de taille {max(1, b_ // 2 - 5)} : "
                      f"jetons identiques {same:.1%}")
                if same < 0.9:
                    raise RuntimeError("la régénération ne redonne pas les mêmes suites : chemin de génération défectueux")
                a["reprise"] = same
            uniq = [len({tuple(r) for r in TK[p]}) / N_ECH for p in range(len(PROMPTS))]
            print(f"[MESURE] {label} T = {Tv} : {time.time() - t1:.0f} s ; part de suites distinctes : médiane {np.median(uniq):.0%}, "
                  f"min {min(uniq):.0%} ; entropie moyenne {S.mean():.3f}", flush=True)
            rT = {"distinctes": uniq, "entropie": float(S.mean())}
            rT["h"] = analyser(H[couches[0]], "B2-h" if ti == 0 else "B2-h, descriptif", label, Tv, SEED + 100 * ti)
            rT["S"] = analyser(S, "B2-S" if ti == 0 else "B2-S, descriptif", label, Tv, SEED + 100 * ti + 50)
            rT["h_finale"] = analyser(H[couches[1]], "couche finale, descriptif", label, Tv, SEED + 100 * ti + 70)
            multi = np.nonzero(rT["h"]["multimodal"])[0]
            ks, ratios, acc1 = [], [], []
            rng_b = np.random.default_rng(SEED + 7)
            for n_ex, p in enumerate(multi):
                lab = etiquettes(H[couches[0]][p].astype(np.float64), int(SEED % 2 ** 31))
                k, acc, has, ratio = bifurcation(TK[p], lab, S[p], rng_b)
                ks.append(k)
                ratios.append(ratio)
                acc1.append(float(acc[0] - has[0]))
                if n_ex < 3 and ti == 0:
                    for c in (0, 1):
                        ex = [tok.decode(TK[p][i].tolist()).replace("\n", " ") for i in np.nonzero(lab == c)[0][:2]]
                        print(f"[EXEMPLE] {label} prompt {p} « {PROMPTS[p]} » bassin {c} ({np.mean(lab == c):.0%}) : "
                              + " | ".join(f"« {e[:90]} »" for e in ex))
            kk = [k for k in ks if k is not None]
            rT["bifurcation"] = {"k": ks, "ratio_entropie": ratios, "gain_premier_jeton": acc1}
            if len(multi):
                print(f"[BIFURCATION] {label} T = {Tv} ({len(multi)} prompts multimodaux) : bifurcation trouvée {len(kk)}/{len(multi)} ; "
                      f"k* médian {np.median(kk) if kk else float('nan'):.0f} ; k* ≤ 4 : {np.mean([k <= 4 for k in kk]) if kk else float('nan'):.0%} ; "
                      f"entropie à k* / autres positions : médiane {np.nanmedian(ratios) if kk else float('nan'):.2f}")
            a[f"T{Tv}"] = rT
            np.savez_compressed(os.path.join(OUT, f"{label}_T{Tv}_b2.npz"), S=S, TK=TK,
                                **{f"H{c}_acp": np.stack([standardiser(H[c][p], H[c][p]) @ acp(standardiser(H[c][p], H[c][p]), 32)
                                                          for p in range(len(PROMPTS))]).astype(np.float32) for c in couches})
            del H
            gc.collect()
        a["B2-h"], a["B2-S"] = a[f"T{TEMPS[0]}"]["h"]["verdict"], a[f"T{TEMPS[0]}"]["S"]["verdict"]
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min", flush=True)
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}", flush=True)
        ALL[label] = {"erreur": f"{type(e).__name__}: {str(e)[:300]}"}
    del model, tok, g
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ecrire_resume()

# %%
# ---------- Synthèse ----------
print(f"\n{'=' * 100}\n[SYNTHÈSE] B2 — les trajectoires d'un même prompt forment-elles des bassins distincts ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "B2-h" in ALL[m]]
for m in labs:
    print(f"[SYNTHÈSE] {m:<22} : B2-h {ALL[m]['B2-h']:<12} ; B2-S {ALL[m]['B2-S']}")
these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese({m: ALL[m]["B2-h"] for m in labs}, ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « les trajectoires d'un même prompt se rangent en bassins distincts » : {these} "
      f"(prédiction SOUTENUE 20 %, RÉFUTÉE 25 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

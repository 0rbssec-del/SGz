# %% [markdown]
# # SGZ A1p — Habituation : un LLM réagit-il moins à une anomalie qu'il a déjà vue ?
#
# **But.** En neurosciences, la réponse cérébrale à une déviance (mismatch negativity) diminue quand la même déviance se
# répète : c'est l'habituation. A1p pose la même question à un LLM : un jeton rare (rang 1000–5000) forcé une **deuxième
# fois** dans le même texte a-t-il un effet direct plus faible sur l'entropie que le même jeton forcé une première fois ?
# Antériorité (recherche économe du 27 oct.) : aucun test trouvé de cette question ; le plus proche, Ren & Wang (COLM 2026,
# arXiv:2608.14681), trouve que la répétition d'un mot ordinaire facilite son traitement de façon stable chez les modèles
# de base, et de façon décroissante, voire en interférence, chez les modèles instruct. Prédiction concurrente : le
# « contextual entrainment » (arXiv:2505.09338) rend un jeton déjà vu plus probable, ce qui peut réduire la surprise sans
# réduire l'effet sur la suite.
#
# **Deux modèles, même famille : Qwen/Qwen3-1.7B-Base et Qwen/Qwen3-1.7B (instruct), texte brut, mêmes 768 prompts,
# T = 0,6.** Premier choc X à t₁ = 64 (exactement comme A1c–A1k : trajectoires du modèle de base identiques à A1k),
# second choc à t₂ = 96, 32 jetons plus tard. Génération libre de deux bras (témoin, choc1) ; tous les autres bras par
# passes rejouées (forçage par l'enseignant), comme le volet « effet direct » d'A1c.
#
# | Bras rejoué | Trajectoire | Jeton imposé à t₂ | Sert à mesurer |
# |---|---|---|---|
# | témoin | témoin | (le sien) | référence |
# | choc1 | choc1 (X à t₁) | (le sien) | référence après le premier choc |
# | double | choc1 | **X** de nouveau | D_rep = effet de X déjà vu |
# | seul2 | témoin | **X** (première fois) | D_seul = effet de X jamais vu, même position |
# | nouveau | choc1 | **Y** : autre jeton, au rang voisin de celui de X à t₂ | D_nouv = effet d'un second choc différent, même rareté |
#
# ## Critères (fixés avant toute exécution)
#
# Effets directs, moyenne sur k = 2–31 après t₂ de la différence d'entropie de Tsallis (q = 1,45) entre bras rejoués :
# D_rep = double − choc1 ; D_seul = seul2 − témoin ; D_nouv = nouveau − choc1. Bootstrap apparié sur les prompts (B = 1000).
# Sont exclus les prompts où X est le jeton que le modèle a lui-même tiré à t₂ (dans le témoin ou dans choc1) : le choc
# n'en est pas un ; la part exclue est imprimée.
# - **V1 habituation**, H = D_rep / D_seul : HABITUATION si IC95 < 1 ; ÉQUIVALENT si IC95 ⊂ [0,8 ; 1,2] ;
#   SENSIBILISATION si IC95 > 1 ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de D_seul contient 0.
# - **V2 spécificité**, Sp = D_rep / D_nouv : SPÉCIFIQUE (l'habituation tient à l'identité du jeton, à rareté égale)
#   si IC95 < 1 ; NON SPÉCIFIQUE si IC95 ⊂ [0,8 ; 1,2] ; INVERSE si IC95 > 1 ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95
#   de D_nouv contient 0.
# - **Descriptifs** : baisse de surprise de X à t₂ (−log p dans le témoin moins −log p après le premier choc) ; effet
#   du premier choc D₁ (k = 2–31 après t₁) ; rapport H(instruct) / H(base).
# - **Synthèse, thèse « un LLM s'habitue à une anomalie répétée »** : SOUTENUE si V1 = HABITUATION pour les deux
#   modèles ; RÉFUTÉE si V1 ∈ {ÉQUIVALENT, SENSIBILISATION} pour les deux ; DÉPEND DU MODÈLE si l'un est HABITUATION et
#   l'autre ÉQUIVALENT ou SENSIBILISATION ; sinon INDÉTERMINÉE.
# - Témoins positifs synthétiques pour chaque règle ; ancrage : trajectoires témoin et choc1 du modèle de base identiques
#   à A1k sur les 96 premiers jetons (MD5).
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1 HABITUATION : base 55 %, instruct 40 % ; SENSIBILISATION 15 % chacun. V2 SPÉCIFIQUE 35 %. H(instruct) > H(base),
# comme le suggère Ren & Wang : 55 %. Synthèse SOUTENUE 30 %, DÉPEND DU MODÈLE 25 %, RÉFUTÉE 20 %.
#
# ## Limites connues
#
# - Effet direct seulement (passes rejouées) : l'effet sur la trajectoire libre après le second choc n'est pas mesuré.
# - Un seul écart entre les chocs (32 jetons), une seule température, une seule famille de modèles.
# - Y est apparié en rareté à X à t₂, pas en sens ni en fréquence d'usage.
# - « Habituation » est une analogie : un transformeur n'a pas de fatigue neuronale ; un effet viendrait du contexte.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~20 à 30 min par
# modèle [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[ANCRAGE]`, `[VÉRIF]`, `[EFFETS]`, `[HABITUATION]`,
# `[SPÉCIFICITÉ]`, `[SYNTHÈSE]`.

# %%
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1p" if os.path.isdir("/kaggle/working") else "sgz_a1p_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T1, T2 = 768, 128, 64, 96
RANK_LO, RANK_HI = 1000, 5000
MID_D = (2, 31)
MARGE = 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                   # même graine que A1c–A1k : mêmes trajectoires jusqu'à t₂
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1K_MD5 = {"Qwen3-1.7B-Base": {"temoin": "cdc160929e703e6ec61452b0c6a742fa", "choc1": "a53ac3b370ec796c8e05a1d5d043115e"}}
BRAS_R = ["temoin", "choc1", "double", "seul2", "nouveau"]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1, T2 = 16, 60, 16, 32
    RANK_LO, RANK_HI = 20, 80
    MID_D = (2, 12)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-aleatoire-bis", ["smoke-qwen3-random"])]
assert T2 + MID_D[1] <= GEN_LEN - 1, "fenêtre après t₂ trop longue pour la génération"
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₁={T1} t₂={T2} fenêtre k={MID_D[0]}–{MID_D[1]} "
      f"marge ±{MARGE:.0%} ; modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# %%
# ---------- Prompts ----------
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
LANG_MAIN = ["fr"] * 32 + ["en"] * 32
THEMES_FR = ["la photosynthèse", "la révolution industrielle", "les volcans", "le jazz", "la cuisine italienne", "les trous noirs",
             "le hockey sur glace", "les abeilles", "l'intelligence artificielle", "le Saint-Laurent", "la Renaissance",
             "les marées", "le vélo en ville", "les vaccins", "la Grèce antique", "les forêts boréales", "le théâtre",
             "l'électricité", "les échecs", "le chocolat", "les ponts suspendus", "la mémoire humaine", "les comètes",
             "le recyclage"]
TECH_FR = ["la chromodynamique quantique", "la catalyse enzymatique", "les réseaux de neurones convolutifs",
           "la cryptographie à clé publique", "la thermodynamique hors équilibre", "la spectroscopie par résonance magnétique nucléaire",
           "la topologie algébrique", "la régulation épigénétique", "les supraconducteurs à haute température",
           "la compilation à la volée", "la turbulence en mécanique des fluides", "l'inférence bayésienne variationnelle",
           "la tectonique des plaques", "les ondes gravitationnelles", "la photolithographie des semi-conducteurs",
           "la théorie des catégories", "l'édition génomique par CRISPR", "les codes correcteurs d'erreurs",
           "la relativité générale", "l'apprentissage par renforcement profond"]
DEBUTS_FR = ["Voici ce qu'il faut savoir sur {t} :", "Hier, un reportage consacré à {t} expliquait que",
             "Pour bien comprendre {t}, il faut d'abord", "L'histoire de {t} commence",
             "Dans ce cours avancé sur {t}, nous montrons que", "Une revue de littérature récente sur {t} conclut que",
             "Le formalisme mathématique de {t} repose sur", "Une erreur fréquente à propos de {t} consiste à"]
THEMES_EN = ["photosynthesis", "the Roman Empire", "earthquakes", "blues music", "Japanese cooking", "neutron stars",
             "basketball", "ants", "machine learning", "the Mississippi River", "the Industrial Revolution", "ocean currents",
             "urban gardening", "antibiotics", "ancient Egypt", "rainforests", "opera", "magnetism", "poker", "coffee",
             "skyscrapers", "human sleep", "asteroids", "solar power"]
TECH_EN = ["quantum chromodynamics", "enzyme catalysis", "convolutional neural networks", "public-key cryptography",
           "non-equilibrium thermodynamics", "nuclear magnetic resonance spectroscopy", "algebraic topology",
           "epigenetic regulation", "high-temperature superconductors", "just-in-time compilation", "turbulent fluid dynamics",
           "variational Bayesian inference", "plate tectonics", "gravitational waves", "semiconductor photolithography",
           "category theory", "CRISPR genome editing", "error-correcting codes", "general relativity",
           "deep reinforcement learning"]
DEBUTS_EN = ["Here is what you need to know about {t}:", "Yesterday, a documentary about {t} explained that",
             "To really understand {t}, you first need to", "The history of {t} begins",
             "In this advanced lecture on {t}, we show that", "A recent literature review on {t} concludes that",
             "The mathematical formalism of {t} relies on", "A common misconception about {t} is that"]


def fr_contract(s):
    """Contractions françaises : « de le » → « du », « à les » → « aux »…"""
    for a, b in ((" de le ", " du "), (" de les ", " des "), (" à le ", " au "), (" à les ", " aux ")):
        s = s.replace(a, b)
    return s


def templates(themes, tech, debuts, lang):
    out = []
    for i, t in enumerate(themes + tech):
        for d in debuts:
            s = d.format(t=t)
            out.append((fr_contract(s) if lang == "fr" else s, lang, i >= len(themes), i - len(themes)))
    return out


TPL_FR, TPL_EN = templates(THEMES_FR, TECH_FR, DEBUTS_FR, "fr"), templates(THEMES_EN, TECH_EN, DEBUTS_EN, "en")
TPL = [x for pair in zip(TPL_FR, TPL_EN) for x in pair]      # français et anglais alternés
ALL_P = [(p, l, False, -1) for p, l in zip(PROMPTS_MAIN, LANG_MAIN)] + TPL
assert len({p for p, *_ in ALL_P}) == len(ALL_P), "prompts en double"
assert not any(x in p for p, *_ in ALL_P for x in (" de le ", " de les ", " à le ", " à les "))
ALL_P = ALL_P[:N_PROMPTS]
PROMPTS = [p for p, *_ in ALL_P]
LANG = [l for _, l, *_ in ALL_P]
print(f"[PROMPTS] {len(PROMPTS)} prompts ({min(64, len(PROMPTS))} écrits à la main, {max(0, len(PROMPTS) - 64)} gabarits), mêmes que A1c–A1k")

# %%
# ---------- Mesures, règles de décision et témoins positifs ----------
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
    """Bruit de Gumbel par ligne, graine explicite par ligne (mêmes nombres pour tous les bras d'un prompt)."""
    g = torch.Generator(device=device)
    rows = []
    for s in seeds:
        g.manual_seed(int(s))
        rows.append(torch.rand(V, generator=g, device=device, dtype=torch.float32))
    u = torch.stack(rows).clamp_(1e-10, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (float("nan"), float("nan"))


def win(X, w):
    return X[:, w[0] - 1:w[1]].mean(1)


def md5(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a).astype(np.int64)).tobytes()).hexdigest()


NOMS = {"H": ("HABITUATION", "ÉQUIVALENT", "SENSIBILISATION"), "Sp": ("SPÉCIFIQUE", "NON SPÉCIFIQUE", "INVERSE")}


def rapport(x, y, boots, cle, interpretable=True, nom_ref="référence"):
    """r = moyenne(x) / moyenne(y), bootstrap apparié sur les prompts ; règles V1 (cle = « H ») et V2 (cle = « Sp »)."""
    bas, egal, haut = NOMS[cle]
    r = float(x.mean() / y.mean())
    c = ci([x[ix].mean() / y[ix].mean() for ix in boots])
    cy = ci([y[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = f"NON TESTABLE ({nom_ref} non significatif)"
    elif c[1] < 1:
        v = bas
    elif c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
        v = egal
    elif c[0] > 1:
        v = haut
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "ci_ref": cy, "verdict": v}


def synthese(v_par_modele, interpretable=True):
    vs = list(v_par_modele.values())
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if vs and all(v == "HABITUATION" for v in vs):
        return "SOUTENUE"
    if vs and all(v in ("ÉQUIVALENT", "SENSIBILISATION") for v in vs):
        return "RÉFUTÉE"
    if "HABITUATION" in vs and any(v in ("ÉQUIVALENT", "SENSIBILISATION") for v in vs):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : rapports connus, bruit multiplicatif par prompt
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
base_pc = rng_pc.lognormal(np.log(0.03), 0.4, Pp)
bruit = lambda f: f * base_pc * rng_pc.lognormal(0, 0.3, Pp) + rng_pc.normal(0, 0.005, Pp)
nul = (lambda x: x - x.mean())(rng_pc.normal(0, 0.01, Pp))
checks = [("habituation (×0,5)", rapport(bruit(0.5), bruit(1.0), bt, "H")["verdict"], "HABITUATION"),
          ("équivalent (×1,0)", rapport(bruit(1.0), bruit(1.0), bt, "H")["verdict"], "ÉQUIVALENT"),
          ("sensibilisation (×1,6)", rapport(bruit(1.6), bruit(1.0), bt, "H")["verdict"], "SENSIBILISATION"),
          ("référence absente", rapport(bruit(1.0), nul, bt, "H", nom_ref="D_seul")["verdict"],
           "NON TESTABLE (D_seul non significatif)"),
          ("spécifique (×0,5)", rapport(bruit(0.5), bruit(1.0), bt, "Sp")["verdict"], "SPÉCIFIQUE"),
          ("non spécifique (×1,0)", rapport(bruit(1.0), bruit(1.0), bt, "Sp")["verdict"], "NON SPÉCIFIQUE"),
          ("synthèse soutenue", synthese({"a": "HABITUATION", "b": "HABITUATION"}), "SOUTENUE"),
          ("synthèse réfutée", synthese({"a": "ÉQUIVALENT", "b": "SENSIBILISATION"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "HABITUATION", "b": "ÉQUIVALENT"}), "DÉPEND DU MODÈLE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<16} (attendu {att})")
POSITIVE_CONTROL_OK = all(g == e for _, g, e in checks)
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON TESTABLE'}")

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


def load_model(model_id, fp32=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == "smoke-qwen3-random":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
        torch.manual_seed(SEED)
        return Qwen3ForCausalLM(cfg).eval().to(DEVICE), SmokeTokenizer(512), "fp32 aléatoire (smoke)"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    elif fp32:
        kw[dkey], kw["device_map"] = torch.float32, ("auto" if N_GPU >= 2 else {"": 0})
        kw["max_memory"] = {i: ("12GiB" if i == 0 else "13GiB") for i in range(N_GPU)}
        mode = f"fp32 sur {N_GPU} GPU"
    elif any(s in model_id for s in ("7B", "8B", "9B", "14B")) and N_GPU >= 2:
        kw[dkey], kw["device_map"] = torch.float16, "auto"
        kw["max_memory"] = {i: ("12GiB" if i == 0 else "13GiB") for i in range(N_GPU)}
        mode = f"fp16 sur {N_GPU} GPU"
    else:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": 0}, "fp16 1 GPU"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda":
        assert next(model.parameters()).dtype == (torch.float32 if fp32 else torch.float16), "type de poids inattendu"
    return model, tok, mode


def set_attention(model, impl):
    """Génération avec l'attention par défaut (comme A1c), passes rejouées en « eager » (poids d'attention visibles)."""
    try:
        model.set_attn_implementation(impl)
    except Exception:
        model.config._attn_implementation = impl
    return getattr(model.config, "_attn_implementation", impl)

# %%
# ---------- Génération (identique à A1c–A1k) ----------
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
        self.t_first = None

    def left_pad(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.dev), attn.to(self.dev), L

    @torch.no_grad()
    def sample(self, ctxs, prompt_of, T, nsteps, step0, force=None):
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
            seeds = [SEED + 100003 * prompt_of[j] + step0 + s for j in range(n)]
            score = z / T + gumbel_noise(self.V, seeds, z.device)
            score[:, self.forbid] = -float("inf")
            if self.V > self.n_tok:
                score[:, self.n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            if force is not None:
                nxt = force(s, z, nxt)
            S[:, s], tk[:, s] = e1, nxt.cpu().numpy()
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = pos[:, -1:] + 1
        del past, out
        return S, tk

    def progress(self, t_lot, n_lots_total):
        if self.t_first is None:
            self.t_first = t_lot
            est = t_lot * n_lots_total / 60
            print(f"[PROGRESSION] {self.label} : premier lot en {t_lot:.1f} s ; estimation ~{est:.0f} min (génération)", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")



# %%
# ---------- Deux chocs : génération et passes rejouées ----------
def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque


@torch.no_grad()
def run_main(g, enc):
    """Témoin et choc1 empilés (comme A1c–A1k : X au rang tiré à t₁, jamais le jeton du témoin). À t₂, rien n'est
    forcé pendant la génération : on note la surprise de X dans les deux trajectoires et on choisit Y, au rang voisin
    de celui de X dans la distribution de choc1."""
    P = len(enc)
    forced_rank = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, P)
    T = T_MAIN[0]
    res = {a: {"S": np.zeros((P, GEN_LEN), np.float32), "tok": np.zeros((P, GEN_LEN), np.int64)} for a in ("temoin", "choc1")}
    info = {k: np.zeros(P, np.int64) for k in ("X", "Y", "rang1", "rangX2", "rangY2")}
    info.update({k: np.zeros(P, np.float64) for k in ("surpX1", "surpX2_temoin", "surpX2_choc1", "surpY2_choc1")})
    npb = max(1, g.budget // 2)
    n_lots = math.ceil(P / npb)
    t0 = time.time()
    for b0 in range(0, P, npb):
        tl = time.time()
        rows_p = list(range(b0, min(P, b0 + npb)))
        n = len(rows_p)

        def force(s, z, nxt, n=n, rows_p=rows_p):
            if s < T1:
                nxt[n:] = nxt[:n]                                      # appariement exact avant le premier choc
            elif s == T1:
                zz = g.masque(z[n:].clone())
                lp = torch.log_softmax(z[n:].double(), dim=-1)
                order = torch.argsort(zz, dim=-1, descending=True)
                for j, r in enumerate(rows_p):
                    rk = int(min(forced_rank[r], order.shape[1] - 2))
                    if int(order[j, rk]) == int(nxt[j]):             # jamais le jeton du témoin
                        rk += 1
                    x = int(order[j, rk])
                    nxt[n + j] = x
                    info["X"][r], info["rang1"][r], info["surpX1"][r] = x, rk, -float(lp[j, x])
            elif s == T2:
                lp = torch.log_softmax(z.double(), dim=-1)
                zz = g.masque(z[n:].clone())
                order = torch.argsort(zz, dim=-1, descending=True)
                for j, r in enumerate(rows_p):
                    x = int(info["X"][r])
                    rx = int((order[j] == x).nonzero()[0, 0])
                    ry = rx + 1 if rx + 1 < order.shape[1] else rx - 1
                    if int(order[j, ry]) == int(nxt[n + j]):         # Y n'est jamais le jeton tiré par choc1
                        ry = ry + 1 if ry + 1 < order.shape[1] else rx - 2
                    y = int(order[j, ry])
                    info["Y"][r], info["rangX2"][r], info["rangY2"][r] = y, rx, ry
                    info["surpX2_temoin"][r] = -float(lp[j, x])
                    info["surpX2_choc1"][r], info["surpY2_choc1"][r] = -float(lp[n + j, x]), -float(lp[n + j, y])
            return nxt

        S, tk = g.sample([enc[r] for r in rows_p] * 2, rows_p * 2, T, GEN_LEN, 0, force)
        for ai, a in enumerate(("temoin", "choc1")):
            sl = slice(ai * n, (ai + 1) * n)
            res[a]["S"][rows_p], res[a]["tok"][rows_p] = S[sl], tk[sl]
        g.progress(time.time() - tl, n_lots)
    ref, fo = res["temoin"]["tok"], res["choc1"]["tok"]
    same = np.array_equal(ref[:, :T1], fo[:, :T1])
    chg = float(np.mean(ref[:, T1] != fo[:, T1]))
    print(f"[VÉRIF] {g.label} : identique avant t₁ : {same} ; jeton changé à t₁ : {chg:.0%} ; rang médian du choc "
          f"{int(np.median(info['rang1']))} ; à t₂, rang médian de X après le premier choc {int(np.median(info['rangX2']))}",
          flush=True)
    if not same or chg < 0.99:
        raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
    print(f"[MESURE] {g.label} génération : {time.time() - t0:.0f} s", flush=True)
    return res, info


@torch.no_grad()
def rejouer(g, ctxs, conts):
    """Entropie de chaque pas d'une suite imposée (forçage par l'enseignant), en une passe sans cache."""
    base, head = g.model.model, g.model.get_output_embeddings()
    n, Kp = len(ctxs), len(conts[0]) + 1
    ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    hk = h[:, L - Kp:, :].reshape(n * Kp, -1)
    flat = np.zeros(n * Kp, np.float64)
    for c0 in range(0, n * Kp, 192):
        flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
    return flat.reshape(n, Kp).astype(np.float32)


@torch.no_grad()
def run_replay(g, enc, res, info):
    P = len(enc)
    SR = {a: np.zeros((P, GEN_LEN), np.float32) for a in BRAS_R}
    nb = 8
    t0 = time.time()
    for b0 in range(0, P, nb):
        rows = list(range(b0, min(P, b0 + nb)))
        ct = [res["temoin"]["tok"][r, :GEN_LEN - 1].tolist() for r in rows]
        cc = [res["choc1"]["tok"][r, :GEN_LEN - 1].tolist() for r in rows]
        sub = lambda c, x: c[:T2] + [int(x)] + c[T2 + 1:]
        conts = {"temoin": ct, "choc1": cc,
                 "double": [sub(c, info["X"][r]) for c, r in zip(cc, rows)],
                 "seul2": [sub(c, info["X"][r]) for c, r in zip(ct, rows)],
                 "nouveau": [sub(c, info["Y"][r]) for c, r in zip(cc, rows)]}
        out = rejouer(g, [enc[r] for r in rows] * len(BRAS_R), [x for a in BRAS_R for x in conts[a]])
        m = len(rows)
        for ai, a in enumerate(BRAS_R):
            SR[a][rows] = out[ai * m:(ai + 1) * m]
    for a in ("temoin", "choc1"):
        d = np.abs(SR[a] - res[a]["S"])
        print(f"[VÉRIF] {g.label} {a} rejoué contre généré : écart moyen {d.mean():.1e}, max {d.max():.1e}", flush=True)
        if d.mean() > 0.1:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
    print(f"[MESURE] {g.label} passes rejouées : {time.time() - t0:.0f} s", flush=True)
    return SR


def effets(SR, t, a, b):
    return win((SR[a] - SR[b])[:, t + 1:].astype(float), MID_D)

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1p.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


PAR = {}
for label, ids in MODELS:
    t_start = time.time()
    model = tok = g = None
    try:
        for mid in ids:
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
        enc = [tok.encode(p)[:12] if SMOKE_TEST else tok.encode(p) for p in PROMPTS]     # comme A1k
        g = Gen(model, tok, label)
        res, info = run_main(g, enc)
        if label in A1K_MD5 and not SMOKE_TEST:
            okt = md5(res["temoin"]["tok"][:, :96]) == A1K_MD5[label]["temoin"]
            okc = md5(res["choc1"]["tok"][:, :96]) == A1K_MD5[label]["choc1"]
            print(f"[ANCRAGE] {label} : 96 premiers jetons identiques à A1k (MD5) : témoin {okt}, choc1 {okc}"
                  + ("" if okt and okc else " (écart numérique possible ; les contrastes, mesurés ici, restent valides)"))
        SR = run_replay(g, enc, res, info)
        valide = (res["temoin"]["tok"][:, T2] != info["X"]) & (res["choc1"]["tok"][:, T2] != info["X"])
        e = {"D1": effets(SR, T1, "choc1", "temoin"), "D_rep": effets(SR, T2, "double", "choc1"),
             "D_seul": effets(SR, T2, "seul2", "temoin"), "D_nouv": effets(SR, T2, "nouveau", "choc1")}
        PAR[label] = {k: v[valide] for k, v in e.items()}
        PAR[label]["dsurp"] = (info["surpX2_temoin"] - info["surpX2_choc1"])[valide]
        P = int(valide.sum())
        rng = np.random.default_rng(SEED + 3)
        boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]
        pv = PAR[label]
        a = {"exclus": float(1 - valide.mean()), "n": P}
        for k in ("D1", "D_rep", "D_seul", "D_nouv", "dsurp"):
            a[k] = {"moyenne": float(pv[k].mean()), "ci": ci([pv[k][ix].mean() for ix in boots])}
        a["surprise"] = {k: float(np.median(info[k][valide])) for k in ("surpX1", "surpX2_temoin", "surpX2_choc1", "surpY2_choc1")}
        a["H"] = rapport(pv["D_rep"], pv["D_seul"], boots, "H", ok, "D_seul")
        a["Sp"] = rapport(pv["D_rep"], pv["D_nouv"], boots, "Sp", ok, "D_nouv")
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts retenus, {a['exclus']:.1%} exclus "
              f"(X tiré par le modèle lui-même à t₂)")
        for k, nom in (("D1", "premier choc, après t₁"), ("D_seul", "X jamais vu, à t₂"), ("D_rep", "X déjà vu, à t₂"),
                       ("D_nouv", "Y nouveau, même rareté, à t₂")):
            print(f"[EFFETS] {k:<6} ({nom:<28}) = {a[k]['moyenne']:+.4f} IC95 [{a[k]['ci'][0]:+.4f} ; {a[k]['ci'][1]:+.4f}]")
        s = a["surprise"]
        print(f"[EFFETS] surprise médiane de X (nats) : à t₁ {s['surpX1']:.2f} ; à t₂ jamais vu {s['surpX2_temoin']:.2f}, "
              f"déjà vu {s['surpX2_choc1']:.2f} ; Y à t₂ {s['surpY2_choc1']:.2f} ; baisse moyenne {a['dsurp']['moyenne']:+.2f} "
              f"IC95 [{a['dsurp']['ci'][0]:+.2f} ; {a['dsurp']['ci'][1]:+.2f}]")
        h, sp = a["H"], a["Sp"]
        print(f"[HABITUATION] {label} : H = D_rep / D_seul = {h['r']:.2f} IC95 [{h['ci'][0]:.2f} ; {h['ci'][1]:.2f}] → {h['verdict']}")
        print(f"[SPÉCIFICITÉ] {label} : Sp = D_rep / D_nouv = {sp['r']:.2f} IC95 [{sp['ci'][0]:.2f} ; {sp['ci'][1]:.2f}] → {sp['verdict']}")
        print(f"[EXTRAIT] {label} prompt 0, double choc : "
              f"{tok.decode(res['choc1']['tok'][0, T1 - 4:T2].tolist() + [int(info['X'][0])] + res['choc1']['tok'][0, T2 + 1:T2 + 12].tolist())!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a1p.npz"), valide=valide,
                            **{f"S_{a_}": res[a_]["S"] for a_ in res}, **{f"tok_{a_}": res[a_]["tok"] for a_ in res},
                            **{f"Srej_{a_}": SR[a_] for a_ in SR}, **{f"info_{k}": v for k, v in info.items()})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1p — habituation à une anomalie répétée")
vH = {m: ALL[m]["H"]["verdict"] for m in ALL if "H" in ALL[m]}
for m, v in vH.items():
    print(f"[SYNTHÈSE] {m:<22} : V1 {v:<16} ; V2 {ALL[m]['Sp']['verdict']}")
if len(vH) < len(MODELS):
    these = "NON TESTABLE (modèle manquant)"
else:
    these = synthese(vH, ok)
ALL["these"] = these
labs = [m for m, _ in MODELS]
if len(labs) == 2 and all(m in PAR for m in labs) and min(len(PAR[m]["D_rep"]) for m in labs) > 0:
    rngS = np.random.default_rng(SEED + 7)
    bA = [rngS.integers(0, len(PAR[labs[0]]["D_rep"]), len(PAR[labs[0]]["D_rep"])) for _ in range(BOOT_N)]
    bB = [rngS.integers(0, len(PAR[labs[1]]["D_rep"]), len(PAR[labs[1]]["D_rep"])) for _ in range(BOOT_N)]
    Hm = lambda m, ix: PAR[m]["D_rep"][ix].mean() / PAR[m]["D_seul"][ix].mean()
    rr = Hm(labs[1], slice(None)) / Hm(labs[0], slice(None))
    c = ci([Hm(labs[1], ib) / Hm(labs[0], ia) for ia, ib in zip(bA, bB)])
    ALL["H_instruct_sur_base"] = {"r": float(rr), "ci": c}
    print(f"[SYNTHÈSE] H({labs[1]}) / H({labs[0]}) = {rr:.2f} IC95 [{c[0]:.2f} ; {c[1]:.2f}] (descriptif ; > 1 : "
          f"le modèle instruct s'habitue moins, comme le suggère Ren & Wang)")
print(f"[SYNTHÈSE] thèse « un LLM s'habitue à une anomalie répétée » : {these} "
      f"(prédiction SOUTENUE 30 %, DÉPEND DU MODÈLE 25 %, RÉFUTÉE 20 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

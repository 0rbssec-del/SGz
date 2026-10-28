# %% [markdown]
# # SGZ A1s — Courbe dose-réponse : l'effet d'une anomalie sature-t-il avec sa surprise ?
#
# **Point de départ (analyse après coup d'A1p, 28 oct.).** Sur Qwen3-1.7B base et instruct, l'effet direct du premier choc
# (rang 1000–5000) ne suit pas sa surprise, qui va de 10 à 48 nats : Spearman +0,05 et −0,01 ; moyennes par quartile non
# croissantes. Lecture proposée : au-delà d'une dizaine de nats, l'intensité de la surprise n'ajoute plus rien (plateau).
# Lecture concurrente : la surprise compte, mais la bande 1000–5000 est trop étroite pour le voir.
#
# **Dispositif.** Témoin seul généré (768 prompts, T = 0,6, 96 jetons, même graine que A1k et A1r), puis passes rejouées
# (forçage par l'enseignant) où seul le jeton t₀ = 64 change, la suite restant celle du témoin. Treize doses : le jeton de
# rang r dans la loi du témoin au pas t₀, r ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 20000} (rang 0 =
# le plus probable ; jamais le jeton tiré par le témoin, on passe alors au rang suivant ; jetons interdits exclus).
# Deux modèles : Qwen3-1.7B-Base et SmolLM3-3B-Base (comme A1r).
#
# ## Mesures
#
# - **D (principale)** : effet direct sur l'entropie de Tsallis (q = 1,45), moyenne de bras − témoin sur k = 2–31.
# - **A (secondaire)** : surplus de part d'attention vers la position t₀ (moyenne sur couches, têtes et k = 2–31), bras −
#   témoin.
# - Surprise de chaque dose : −log p du jeton forcé dans la loi du témoin (T = 1), médiane par dose.
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# Pour chaque mesure, M(r) = moyenne sur les prompts de la dose r.
# - **V1, croissance en zone basse** : M(50) / M(2) : CROÎT si IC95 > 1 ; NE CROÎT PAS si IC95 ⊂ [0,8 ; 1,2] ou < 1 ;
#   sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de M(2) contient 0. (Contrôle : sans cela, le plateau ne veut rien dire.)
# - **V2, plateau en zone haute** : P = M(20000) / M(1000), sur un facteur 20 de rang : PLATEAU si IC95 ⊂ [0,8 ; 1,2] ;
#   sinon CROISSANCE CONTINUE si IC95 > 1 ; DÉCROISSANCE si IC95 < 1 ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de M(1000)
#   contient 0.
# - **Synthèse, thèse « l'effet d'une anomalie sature au-delà d'une surprise d'une dizaine de nats »**, sur D : SOUTENUE si
#   V1 = CROÎT et V2 = PLATEAU pour les deux modèles ; RÉFUTÉE si V2 = CROISSANCE CONTINUE pour les deux ; DÉPEND DU
#   MODÈLE si l'un est PLATEAU (avec V1 = CROÎT) et l'autre CROISSANCE CONTINUE ; sinon INDÉTERMINÉE. Même synthèse,
#   secondaire, sur A.
# - **Descriptifs** : courbe M(r) avec IC95 et surprise médiane par dose ; seuil de saturation = plus petit rang r tel que
#   M(r) ≥ 0,8 × M(20000) ; pente de M contre la surprise par segment ; Spearman par prompt entre M et la surprise dans
#   la zone haute (r ≥ 1000) ; valeurs à k = 1.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1(D) CROÎT : 90 % par modèle. V2(D) : PLATEAU 55 %, CROISSANCE CONTINUE 25 %, DÉCROISSANCE 10 %. Synthèse sur D :
# SOUTENUE 45 %, RÉFUTÉE 20 %. Seuil de saturation entre les rangs 50 et 1000 : 60 %.
#
# ## Limites connues
#
# - Le rang est la dose ; la surprise qui lui correspond varie d'un prompt à l'autre (contexte plus ou moins piqué).
# - Les jetons des fortes doses diffèrent aussi en nature (fragments, autres écritures) : un plateau peut mêler surprise et
#   nature du jeton (voir A1r pour l'écriture).
# - Effet direct par passes rejouées ; une température ; deux modèles de 1,7B et 3B.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~25 min pour Qwen3,
# ~45 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[COURBE]`, `[V1]`, `[V2]`,
# `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1s" if os.path.isdir("/kaggle/working") else "sgz_a1s_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 768, 96, 64
RANGS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 20000]
R_BAS, R_MILIEU, R_HAUT, R_SOMMET = 2, 50, 1000, 20000      # V1 : M(50)/M(2) ; V2 : M(20000)/M(1000)
MID = (2, 31)
MARGE = 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1r : même témoin
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1K_MD5 = {"Qwen3-1.7B-Base": "cdc160929e703e6ec61452b0c6a742fa"}   # témoin d'A1k, 768 × 96 jetons
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 16, 48, 12
    RANGS = [1, 2, 5, 10, 20, 50, 80]
    R_BAS, R_MILIEU, R_HAUT, R_SOMMET = 2, 10, 20, 80
    MID = (2, 12)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-aleatoire-bis", ["smoke-qwen3-random"])]
BRAS = ["temoin"] + [f"r{r}" for r in RANGS]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
assert T0 + MID[1] <= GEN_LEN - 1 and {R_BAS, R_MILIEU, R_HAUT, R_SOMMET} <= set(RANGS)
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T0} fenêtre k={MID[0]}–{MID[1]} ; rangs {RANGS} ; "
      f"modèles={[m[0] for m in MODELS]}")
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


def rapport_ci(x, y, boots):
    return float(x.mean() / y.mean()), ci([x[ix].mean() / y[ix].mean() for ix in boots]), ci([y[ix].mean() for ix in boots])


def regle_v1(x, y, boots, interpretable=True):
    """Croissance en zone basse : x = M(50), y = M(2)."""
    r, c, cy = rapport_ci(x, y, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (M(2) non significatif)"
    elif c[0] > 1:
        v = "CROÎT"
    elif c[1] < 1 or (c[0] >= 1 - MARGE and c[1] <= 1 + MARGE):
        v = "NE CROÎT PAS"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def regle_v2(x, y, boots, interpretable=True):
    """Plateau en zone haute : x = M(20000), y = M(1000)."""
    r, c, cy = rapport_ci(x, y, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (M(1000) non significatif)"
    elif c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
        v = "PLATEAU"
    elif c[0] > 1:
        v = "CROISSANCE CONTINUE"
    elif c[1] < 1:
        v = "DÉCROISSANCE"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def synthese(v1, v2, interpretable=True):
    """v1, v2 : dict modèle → verdict."""
    if not interpretable:
        return "NON INTERPRÉTABLE"
    ms = list(v2)
    if not ms:
        return "INDÉTERMINÉE"
    plat = [m for m in ms if v2[m] == "PLATEAU" and v1.get(m) == "CROÎT"]
    cont = [m for m in ms if v2[m] == "CROISSANCE CONTINUE"]
    if len(plat) == len(ms):
        return "SOUTENUE"
    if len(cont) == len(ms):
        return "RÉFUTÉE"
    if plat and cont and len(plat) + len(cont) == len(ms):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


def seuil(Mr):
    """Plus petit rang r tel que M(r) ≥ 0,8 × M(rang maximal) ; Mr : dict rang → moyenne."""
    top = Mr[RANGS[-1]]
    for r in RANGS:
        if Mr[r] >= 0.8 * top:
            return r
    return None


# Témoins positifs : courbes connues en fonction de la surprise s(r), bruit multiplicatif par prompt
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
sens = rng_pc.lognormal(np.log(0.05), 0.5, Pp)
s_de = {r: 1.5 + 1.4 * math.log(1 + r) for r in RANGS}     # surprise croissante avec le rang (forme indicative)


def courbe(f):
    return {r: f(s_de[r]) * sens * rng_pc.lognormal(0, 0.15, Pp) for r in RANGS}


plateau = courbe(lambda s: 1 - math.exp(-s / 3))
lineaire = courbe(lambda s: s)
plat_partout = courbe(lambda s: 1.0)
baisse = courbe(lambda s: 1 / s)
checks = [("croissance basse", regle_v1(plateau[R_MILIEU], plateau[R_BAS], bt)["verdict"], "CROÎT"),
          ("pas de croissance", regle_v1(plat_partout[R_MILIEU], plat_partout[R_BAS], bt)["verdict"], "NE CROÎT PAS"),
          ("plateau", regle_v2(plateau[R_SOMMET], plateau[R_HAUT], bt)["verdict"], "PLATEAU"),
          ("croissance continue", regle_v2(lineaire[R_SOMMET], lineaire[R_HAUT], bt)["verdict"], "CROISSANCE CONTINUE"),
          ("décroissance", regle_v2(baisse[R_SOMMET], baisse[R_HAUT], bt)["verdict"], "DÉCROISSANCE"),
          ("synthèse soutenue", synthese({"a": "CROÎT", "b": "CROÎT"}, {"a": "PLATEAU", "b": "PLATEAU"}), "SOUTENUE"),
          ("plateau sans croissance", synthese({"a": "NE CROÎT PAS", "b": "CROÎT"}, {"a": "PLATEAU", "b": "PLATEAU"}),
           "INDÉTERMINÉE"),
          ("synthèse réfutée", synthese({"a": "CROÎT", "b": "CROÎT"},
                                        {"a": "CROISSANCE CONTINUE", "b": "CROISSANCE CONTINUE"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "CROÎT", "b": "CROÎT"}, {"a": "PLATEAU", "b": "CROISSANCE CONTINUE"}),
           "DÉPEND DU MODÈLE"),
          ("seuil", str(seuil({r: 1 - math.exp(-s_de[r] / 3) for r in RANGS})), "10")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<20} (attendu {att})")
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
# ---------- Écriture des jetons, choix des quatre bras ----------
def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque


# %%
# ---------- Passes rejouées : entropie et part d'attention vers t₀ ----------
@torch.no_grad()
def rejouer(g, ctxs, conts):
    """Pour chaque suite imposée : entropie aux indices t₀ … t₀ + 31 (indice i : loi du jeton i) et part d'attention de la
    position qui prédit le jeton i vers la position du jeton t₀ (moyenne sur couches et têtes)."""
    base, head = g.model.model, g.model.get_output_embeddings()
    n, Kp = len(ctxs), len(conts[0]) + 1
    ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    out = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False, output_attentions=True)
    q0, kcol = L - Kp + T0, L - Kp + T0 + 1                   # requête de l'indice t₀ ; clé : le jeton t₀
    att = out.attentions
    assert att is not None and att[0] is not None, "poids d'attention indisponibles"
    A = torch.stack([a[:, :, q0:q0 + NPOS, kcol].float().mean(1).to(ids.device) for a in att]).mean(0).cpu().numpy()
    hk = out.last_hidden_state[:, q0:q0 + NPOS, :].reshape(n * NPOS, -1)
    flat = np.zeros(n * NPOS, np.float64)
    for c0 in range(0, n * NPOS, 192):
        flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
    del out, att
    return flat.reshape(n, NPOS).astype(np.float32), A.astype(np.float32)


@torch.no_grad()
def generer_temoin(g, enc):
    """Témoin seul (même graine par prompt que A1c–A1r) ; au pas t₀, jeton de chaque rang dans la loi du témoin."""
    P = len(enc)
    tokm, Sg = np.zeros((P, GEN_LEN), np.int64), np.zeros((P, GEN_LEN), np.float32)
    jet, rg, surp = (np.zeros((P, len(RANGS)), t) for t in (np.int64, np.int64, np.float64))
    npb = g.budget
    n_lots = math.ceil(P / npb)
    t0 = time.time()
    for b0 in range(0, P, npb):
        tl = time.time()
        rows = list(range(b0, min(P, b0 + npb)))

        def force(s, z, nxt, rows=rows):
            if s == T0:
                zz = g.masque(z.clone())
                lp = torch.log_softmax(z.double(), dim=-1)
                order = torch.argsort(zz, dim=-1, descending=True)
                for j, r in enumerate(rows):
                    pos_own = int((order[j] == nxt[j]).nonzero()[0, 0])
                    for i, rr in enumerate(RANGS):
                        rk = rr + (1 if pos_own <= rr else 0)          # jamais le jeton du témoin
                        x = int(order[j, rk])
                        jet[r, i], rg[r, i], surp[r, i] = x, rk, -float(lp[j, x])
            return nxt

        S, tk = g.sample([enc[r] for r in rows], rows, T_MAIN[0], GEN_LEN, 0, force)
        tokm[rows], Sg[rows] = tk, S
        g.progress(time.time() - tl, n_lots)
    print(f"[MESURE] {g.label} génération du témoin : {time.time() - t0:.0f} s", flush=True)
    return tokm, Sg, jet, rg, surp


@torch.no_grad()
def run_replay(g, enc, tokm, jet):
    P = len(enc)
    SR = {a: np.zeros((P, NPOS), np.float32) for a in BRAS}
    AR = {a: np.zeros((P, NPOS), np.float32) for a in BRAS}
    nb = 3
    t0 = time.time()
    for b0 in range(0, P, nb):
        rows = list(range(b0, min(P, b0 + nb)))
        ct = [tokm[r, :T0 + NPOS - 1].tolist() for r in rows]
        conts = {"temoin": ct}
        for i, a in enumerate(BRAS[1:]):
            conts[a] = [c[:T0] + [int(jet[r, i])] + c[T0 + 1:] for c, r in zip(ct, rows)]
        S, A = rejouer(g, [enc[r] for r in rows] * len(BRAS), [x for a in BRAS for x in conts[a]])
        m = len(rows)
        for ai, a in enumerate(BRAS):
            SR[a][rows], AR[a][rows] = S[ai * m:(ai + 1) * m], A[ai * m:(ai + 1) * m]
        if b0 == 0:
            est = (time.time() - t0) * P / nb / 60
            print(f"[PROGRESSION] {g.label} passes rejouées : estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
    print(f"[MESURE] {g.label} passes rejouées : {time.time() - t0:.0f} s", flush=True)
    return SR, AR

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1s.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


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
        tokm, Sg, jet, rg, surp = generer_temoin(g, enc)
        if label in A1K_MD5 and not SMOKE_TEST:
            okm = md5(tokm[:, :96]) == A1K_MD5[label]
            print(f"[ANCRAGE] {label} : témoin identique à A1k (MD5, 768 × 96) : {okm}"
                  + ("" if okm else " (écart numérique possible ; les contrastes, mesurés ici, restent valides)"))
        impl = set_attention(model, "eager")
        print(f"[CONFIG] {label} : passes rejouées en attention « {impl} » (poids d'attention lus)")
        SR, AR = run_replay(g, enc, tokm, jet)
        P = len(enc)
        dd = np.abs(SR["temoin"] - Sg[:, T0:T0 + NPOS])
        av = max(float(np.abs(AR[b][:, 0]).max()) for b in BRAS)
        sv = max(float(np.abs(SR[b][:, 0] - SR["temoin"][:, 0]).max()) for b in BRAS)
        print(f"[VÉRIF] {label} témoin rejoué contre généré : écart moyen {dd.mean():.1e}, max {dd.max():.1e} ; avant t₀, "
              f"attention vers t₀ (doit être 0, masque causal) : max {av:.1e} ; entropie identique entre bras : max {sv:.1e}")
        if dd.mean() > 0.1 or av > 1e-3 or sv > 0.05:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
        w = slice(MID[0], MID[1] + 1)
        M = {"D": {r: (SR[f"r{r}"] - SR["temoin"])[:, w].mean(1).astype(float) for r in RANGS},
             "A": {r: (AR[f"r{r}"] - AR["temoin"])[:, w].mean(1).astype(float) for r in RANGS}}
        K1 = {r: (SR[f"r{r}"] - SR["temoin"])[:, 1].astype(float) for r in RANGS}
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        smed = {r: float(np.median(surp[:, i])) for i, r in enumerate(RANGS)}
        a = {"surprise_mediane": smed, "rang_median": {r: float(np.median(rg[:, i])) for i, r in enumerate(RANGS)}}
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts ; part d'attention vers t₀ dans le témoin "
              f"(k = 2–31) {AR['temoin'][:, w].mean():.4f}")
        for m in ("D", "A"):
            Mr = {r: float(M[m][r].mean()) for r in RANGS}
            for i, r in enumerate(RANGS):
                c_ = ci([M[m][r][ix].mean() for ix in boots])
                extra = f" ; à k = 1 {K1[r].mean():+.4f}" if m == "D" else ""
                print(f"[COURBE] {label} {m} rang {r:>5} (surprise médiane {smed[r]:5.2f} nats) : {Mr[r]:+.5f} IC95 "
                      f"[{c_[0]:+.5f} ; {c_[1]:+.5f}]{extra}")
            v1 = regle_v1(M[m][R_MILIEU], M[m][R_BAS], boots, ok)
            v2 = regle_v2(M[m][R_SOMMET], M[m][R_HAUT], boots, ok)
            se = seuil(Mr)
            hauts = [i for i, r in enumerate(RANGS) if r >= R_HAUT]
            x = np.concatenate([surp[:, i] for i in hauts])
            y = np.concatenate([M[m][RANGS[i]] for i in hauts])
            rho = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
            pentes = [(RANGS[i], RANGS[i + 1], (Mr[RANGS[i + 1]] - Mr[RANGS[i]]) / max(smed[RANGS[i + 1]] - smed[RANGS[i]], 1e-9))
                      for i in range(len(RANGS) - 1)]
            print(f"[V1] {label} {m} : M({R_MILIEU}) / M({R_BAS}) = {v1['r']:.2f} IC95 [{v1['ci'][0]:.2f} ; {v1['ci'][1]:.2f}] → {v1['verdict']}")
            print(f"[V2] {label} {m} : M({R_SOMMET}) / M({R_HAUT}) = {v2['r']:.2f} IC95 [{v2['ci'][0]:.2f} ; {v2['ci'][1]:.2f}] → {v2['verdict']}")
            print(f"[COURBE] {label} {m} : seuil de saturation (M ≥ 0,8 × M({RANGS[-1]})) au rang {se} "
                  f"(surprise médiane {smed[se] if se else float('nan'):.2f} nats) ; Spearman par prompt en zone haute "
                  f"(rang ≥ {R_HAUT}) {rho:+.3f}")
            print(f"[COURBE] {label} {m} : pente par nat de surprise, segment par segment : "
                  + " ; ".join(f"{r0}→{r1} {p:+.4f}" for r0, r1, p in pentes))
            a[m] = {"courbe": Mr, "V1": v1, "V2": v2, "seuil": se, "spearman_haut": rho, "pentes": pentes}
        a["k1"] = {r: float(K1[r].mean()) for r in RANGS}
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1s.npz"), tok=tokm, jeton=jet, rang=rg, surprise=surp,
                            **{f"S_{b}": SR[b] for b in BRAS}, **{f"A_{b}": AR[b] for b in BRAS})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1s — l'effet d'une anomalie sature-t-il avec sa surprise ?")
for m in ("D", "A"):
    labs = [lab for lab in ALL if isinstance(ALL[lab], dict) and m in ALL[lab]]
    v1 = {lab: ALL[lab][m]["V1"]["verdict"] for lab in labs}
    v2 = {lab: ALL[lab][m]["V2"]["verdict"] for lab in labs}
    for lab in labs:
        print(f"[SYNTHÈSE] {lab:<18} {m} : V1 {v1[lab]:<14} ; V2 {v2[lab]:<20} ; seuil au rang {ALL[lab][m]['seuil']}")
    these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese(v1, v2, ok)
    ALL[f"these_{m}"] = these
    print(f"[SYNTHÈSE] thèse « l'effet sature au-delà d'une dizaine de nats », mesure {m}"
          f"{' (principale)' if m == 'D' else ' (secondaire)'} : {these}"
          + (" (prédiction SOUTENUE 45 %, RÉFUTÉE 20 %)" if m == "D" else ""))
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

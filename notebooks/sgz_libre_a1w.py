# %% [markdown]
# # SGZ A1w — Choc en génération libre : l'effet d'un jeton plausible vient-il de la suite rejouée ?
#
# **Point de départ.** Deux résultats reposent sur des passes rejouées, où l'on remplace le jeton t₀ en gardant la suite
# du témoin, écrite par le modèle pour un autre jeton :
# - A1s : l'effet direct D vaut déjà 68 à 77 % de sa valeur au rang 5000 pour le jeton de rang 1, un jeton plausible
#   (3 nats) ;
# - A1v, après coup : à surprise égale, un jeton inséré reçoit plus d'attention qu'un jeton produit par le modèle (AUC 0,63
#   à 0,75).
#
# Lecture concurrente commune : la suite rejouée ne colle plus au jeton remplacé, et c'est ce désaccord, pas le jeton, qui
# fait monter l'entropie et l'attention. La génération libre le supprime : après le jeton forcé, le modèle écrit lui-même
# la suite.
#
# **Dispositif.** Témoin d'A1s (768 prompts, T = 0,6, 96 jetons, même graine ; ancrage MD5 sur A1s). Au pas t₀ = 64, six
# doses : le jeton de rang r ∈ {1, 5, 50, 1000, 5000, 20000} dans la loi du témoin, choisi comme dans A1s (rang 0 = le plus
# probable ; jamais le jeton du témoin ; ancrage MD5 des jetons). Pour chaque dose, deux textes :
# - **rejoué** : jeton r à t₀, suite du témoin (le bras d'A1s) ;
# - **libre** : jeton r à t₀, puis 30 jetons écrits par le modèle, avec à chaque position le même bruit de Gumbel que le
#   témoin (nombres aléatoires communs, comme le bras choc d'A1c ; la loi de la suite n'en dépend pas, seule la variance
#   des écarts baisse).
#
# Bras de contrôle :
# - **nul** : jeton du témoin à t₀, suite libre avec un autre bruit (effet attendu : 0) ;
# - **reprise** : jeton du témoin, même bruit ; doit redonner le témoin (contrôle du chemin de génération) ;
# - **libre indépendant** (doses 1 et 5000) : suite libre avec un bruit propre, sans couplage au témoin ;
# - **inverse** (doses 1 et 5000) : jeton du témoin remis à t₀, devant la suite libre indépendante écrite pour le jeton r.
#   C'est le miroir du rejoué : une suite écrite pour un autre jeton, mais le jeton inséré est cette fois le plus probable.
#
# Tous les textes passent ensuite en passe rejouée (forçage par l'enseignant, attention « eager ») : entropie de Tsallis
# (q = 1,45) aux indices t₀ … t₀ + 31 et attention reçue par t₀. Deux modèles : Qwen3-1.7B-Base et SmolLM3-3B-Base (comme
# A1s et A1v).
#
# ## Mesures (par prompt, moyenne sur k = 2–31 ; indice k : loi du jeton t₀ + k)
#
# - D(r) = S(rejoué) − S(témoin) : effet direct d'A1s.
# - T(r) = S(libre) − S(témoin) : effet total en génération libre.
# - I(r) = T(r) − D(r) = S(libre) − S(rejoué) : effet de la trajectoire de A1c (règle F).
# - D_inv(r) = S(inverse) − S(libre indépendant) : effet d'une suite étrangère derrière le jeton le plus probable.
# - r₁ = attention de la position t₀ + 1 vers t₀ (moyenne sur couches et têtes, celle d'A1v) ; excès E = r₁(bras) −
#   r₁(témoin).
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# - **W1 (principal), rang 1** : ρ = T(1) / D(1). NON TESTABLE si l'IC95 de D(1) contient 0 ; GONFLÉ PAR LE REJEU si la
#   borne haute de ρ < 0,5 ; PARTIELLEMENT GONFLÉ si IC95 ⊂ [0,5 ; 0,8[ ; CONSERVÉ si la borne basse ≥ 0,8 ; sinon
#   INDÉTERMINÉ.
# - **W2, symétrie au rang 1** : σ = D_inv(1) / D(1). SYMÉTRIQUE si IC95 ⊂ [0,8 ; 1,25] (l'effet ne dépend que du désaccord
#   avec la suite, pas du jeton inséré) ; LE JETON COMPTE si la borne haute < 0,8 ; INVERSE PLUS FORT si la borne basse >
#   1,25 ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de D(1) contient 0.
# - **W3, forme de la courbe libre** : P = T(20000) / T(1000), règle V2 d'A1s : PLATEAU si IC95 ⊂ [0,8 ; 1,2] ; sinon
#   CROISSANCE CONTINUE si la borne basse > 1 ; DÉCROISSANCE si la borne haute < 1 ; sinon INDÉTERMINÉ ; NON TESTABLE si
#   l'IC95 de T(1000) contient 0. La même règle sur D doit redonner A1s (PLATEAU).
# - **W4, attention, rang 50** (jeton plausible, comme A1v) : φ = E_libre(50) / E_rejoué(50). LIÉ À LA SUITE INCOHÉRENTE si
#   la borne haute < 0,5 ; PARTIEL si IC95 ⊂ [0,5 ; 0,8[ ; PORTÉ PAR LE JETON si la borne basse ≥ 0,8 ; sinon INDÉTERMINÉ ;
#   NON TESTABLE si l'IC95 de E_rejoué(50) contient 0.
# - **Témoin négatif** : T(nul) doit être équivalent à 0, IC95 ⊂ [−0,3 × D(1000) ; +0,3 × D(1000)]. Sinon W1 à W4 sont
#   NON INTERPRÉTABLES : la génération libre serait biaisée.
# - **Synthèse 1, thèse « dans une passe rejouée, l'effet d'un jeton plausible vient surtout du désaccord avec la
#   suite »** : SOUTENUE si W1 = GONFLÉ PAR LE REJEU et W2 = SYMÉTRIQUE pour les deux modèles ; PARTIELLE si W1 ∈ {GONFLÉ PAR
#   LE REJEU, PARTIELLEMENT GONFLÉ} pour les deux sans être SOUTENUE ; RÉFUTÉE si W1 = CONSERVÉ pour les deux ; DÉPEND DU
#   MODÈLE si l'un est gonflé (entièrement ou partiellement) et l'autre CONSERVÉ ; sinon INDÉTERMINÉE.
# - **Synthèse 2, thèse « l'excès d'attention vers un jeton inséré plausible tient au désaccord avec la suite »** (W4) :
#   SOUTENUE si LIÉ À LA SUITE INCOHÉRENTE pour les deux ; RÉFUTÉE si PORTÉ PAR LE JETON pour les deux ; DÉPEND DU MODÈLE
#   si l'un est LIÉ et l'autre PORTÉ ; sinon INDÉTERMINÉE.
# - **Synthèse 3, thèse « en génération libre, l'effet croît encore avec la surprise au-delà du rang 1000 »** (W3) :
#   SOUTENUE si CROISSANCE CONTINUE pour les deux ; RÉFUTÉE si PLATEAU ou DÉCROISSANCE pour les deux ; DÉPEND DU MODÈLE si
#   l'un est CROISSANCE CONTINUE et l'autre PLATEAU ou DÉCROISSANCE ; sinon INDÉTERMINÉE.
# - **Descriptifs** : courbes D, T, I par rang avec IC95 ; règle F d'A1c à chaque rang (reproduction attendue :
#   AMPLIFICATION aux rangs 1000 et 5000) ; profils par tranches de k ; part de la suite libre identique au témoin ;
#   T du libre couplé contre T du libre indépendant (doivent coïncider) ; D_inv(5000) ; E et φ à tous les rangs ; excès
#   d'attention à surprise et entropie égales, dans le rejoué et dans le libre (comme l'analyse après coup d'A1v) ;
#   ancrage de D sur A1s.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# W1 : GONFLÉ PAR LE REJEU 55 %, PARTIELLEMENT GONFLÉ 25 %, CONSERVÉ 10 %. W2 : SYMÉTRIQUE 45 %, LE JETON COMPTE 35 %.
# W3 : CROISSANCE CONTINUE 55 %, PLATEAU 30 %. W4 : LIÉ 35 %, PARTIEL 20 %, PORTÉ PAR LE JETON 35 %. Règle F :
# AMPLIFICATION aux rangs 1000 et 5000 85 % ; COMPENSATION au rang 1 55 %. Synthèse 1 : SOUTENUE 35 %, PARTIELLE 30 %.
#
# ## Limites connues
#
# - W1 et la règle F décrivent le même fait sous deux noms : T(1) < D(1) s'écrit aussi I(1) < 0 (« compensation »). W2
#   sert à départager les lectures (désaccord symétrique ou effet propre du jeton).
# - T compare deux textes différents ; le couplage du bruit réduit la variance sans changer l'espérance (vérifié par le
#   libre indépendant aux rangs 1 et 5000).
# - Un seul t₀ (64), une température, deux modèles de 1,7B et 3B ; anomalies artificielles (jeton forcé).
# - Libre indépendant et inverse seulement aux rangs 1 et 5000 (coût).
# - Antériorité : l'écart entre forçage par l'enseignant et génération libre est connu côté entraînement (« biais
#   d'exposition », Ranzato et al., 2016 ; Bengio et al., 2015) ; pour la mesure d'effets par passes rejouées, aucune
#   recherche faite pour A1w (non vérifié).
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~16 min pour Qwen3,
# ~25 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[ANCRAGE]`, `[VÉRIF]`,
# `[TÉMOIN NÉGATIF]`, `[W1]` à `[W4]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1w" if os.path.isdir("/kaggle/working") else "sgz_a1w_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 768, 96, 64
RANGS = [1, 5, 50, 1000, 5000, 20000]            # sous-ensemble de la grille d'A1s : mêmes jetons
R_W1, R_W4, R_HAUT, R_SOMMET = 1, 50, 1000, 20000
R_INV = [1, 5000]                                # libre indépendant et inverse
MID = (2, 31)
MARGE = 0.2                                      # règle V2 d'A1s
MARGE_REL = 0.3                                  # règle F d'A1c et témoin négatif
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1v : même témoin
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
# Ancrage sur A1s (fichiers .npz du 28 oct.) : MD5 du témoin (768 × 96), des jetons des six doses, et D par dose
A1S = {"Qwen3-1.7B-Base": {"tok": "6df145ffd4fd77ee3aad005cc15feff1", "jet": "1727c62543952eb4d6f9aecd75ef7574",
                           "D": {1: 0.05210, 5: 0.05919, 50: 0.06847, 1000: 0.07512, 5000: 0.07652, 20000: 0.07381}},
       "SmolLM3-3B-Base": {"tok": "31a0515328597e205de2ab949d133f27", "jet": "2d5372081f631c5316898a2b4ffa994b",
                           "D": {1: 0.05755, 5: 0.06499, 50: 0.07273, 1000: 0.07747, 5000: 0.07430, 20000: 0.06966}}}
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 16, 48, 12
    RANGS = [1, 5, 10, 20, 40, 80]
    R_W1, R_W4, R_HAUT, R_SOMMET = 1, 10, 20, 80
    R_INV = [1, 40]
    MID = (2, 12)
    BOOT_N = 50
    N_PROMPTS = int(os.environ.get("SMOKE_N", "16"))
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
NSUITE = NPOS - 2                                # jetons écrits après t₀ : t₀ + 1 … t₀ + 30
LONG = T0 + 1 + NSUITE                           # longueur des textes rejoués : jetons 0 … t₀ + 30
POOL = (max(4, T0 - 24), min(T0 + 24, LONG - 2))  # jetons du témoin servant à l'appariement (t₀ exclu)
FRAIS = {"nul": SEED + 7919, **{f"libF{r}": SEED + 7919 * (2 + i) for i, r in enumerate(R_INV)}}   # bruits propres
BRAS = (["temoin"] + [f"rej{r}" for r in RANGS] + [f"lib{r}" for r in RANGS] + ["nul"]
        + [f"libF{r}" for r in R_INV] + [f"inv{r}" for r in R_INV])
LIBRES = [f"lib{r}" for r in RANGS] + ["nul"] + [f"libF{r}" for r in R_INV]
assert LONG <= GEN_LEN and {R_W1, R_W4, R_HAUT, R_SOMMET} <= set(RANGS) and R_W1 in R_INV and set(R_INV) <= set(RANGS)
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T0} fenêtre k={MID[0]}–{MID[1]} ; rangs {RANGS} ; "
      f"suite libre {NSUITE} jetons ; bras {len(BRAS)} ; modèles={[m[0] for m in MODELS]}")
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


def md5(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a).astype(np.int64)).tobytes()).hexdigest()


def rapport(x, y, boots):
    """Rapport des moyennes x̄ / ȳ, IC95 du rapport et IC95 de ȳ (mêmes rééchantillons)."""
    return float(x.mean() / y.mean()), ci([x[ix].mean() / y[ix].mean() for ix in boots]), ci([y[ix].mean() for ix in boots])


NOMS_W1 = ("GONFLÉ PAR LE REJEU", "PARTIELLEMENT GONFLÉ", "CONSERVÉ")
NOMS_W4 = ("LIÉ À LA SUITE INCOHÉRENTE", "PARTIEL", "PORTÉ PAR LE JETON")


def regle_part(x, y, boots, noms, interpretable=True):
    """Part de l'effet y qui subsiste dans x (W1, W4) : borne haute < 0,5 → noms[0] ; IC ⊂ [0,5 ; 0,8[ → noms[1] ;
    borne basse ≥ 0,8 → noms[2]."""
    r, c, cy = rapport(x, y, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (effet du rejeu non significatif)"
    elif c[1] < 0.5:
        v = noms[0]
    elif c[0] >= 0.5 and c[1] < 0.8:
        v = noms[1]
    elif c[0] >= 0.8:
        v = noms[2]
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def regle_sym(x, y, boots, interpretable=True):
    """W2 : σ = D_inv / D."""
    r, c, cy = rapport(x, y, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (effet du rejeu non significatif)"
    elif c[0] >= 0.8 and c[1] <= 1.25:
        v = "SYMÉTRIQUE"
    elif c[1] < 0.8:
        v = "LE JETON COMPTE"
    elif c[0] > 1.25:
        v = "INVERSE PLUS FORT"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def regle_v2(x, y, boots, interpretable=True):
    """Règle V2 d'A1s : x = M(20000), y = M(1000)."""
    r, c, cy = rapport(x, y, boots)
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


def regle_nul(x, marge, boots, interpretable=True):
    """Témoin négatif : moyenne de x équivalente à 0 à la marge près."""
    c = ci([x[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif -marge <= c[0] and c[1] <= marge:
        v = "ÉQUIVALENT À 0"
    elif c[0] > 0 or c[1] < 0:
        v = "BIAISÉ"
    else:
        v = "IMPRÉCIS"
    return {"m": float(x.mean()), "ci": c, "marge": marge, "verdict": v}


def regle_F(Dp, Ip, boots, interpretable=True):
    """Règle F de A1c : Dp, Ip = effet direct et effet de la trajectoire par prompt (moyennes sur k = 2–31)."""
    d, i = float(Dp.mean()), float(Ip.mean())
    dci, ici = ci([Dp[ix].mean() for ix in boots]), ci([Ip[ix].mean() for ix in boots])
    m = MARGE_REL * abs(d)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif dci[0] <= 0 <= dci[1]:
        v = "NON TESTABLE (effet direct absent)"
    elif -m <= ici[0] and ici[1] <= m:
        v = "ÉQUIVALENT"
    elif ici[1] < 0:
        v = "COMPENSATION"
    elif ici[0] > 0:
        v = "AMPLIFICATION"
    else:
        v = "INDÉTERMINÉ"
    return {"D": d, "D_ci": dci, "I": i, "I_ci": ici, "I_sur_D": i / d if d else float("nan"), "verdict": v}


def synthese1(w1, w2, interpretable=True):
    """w1, w2 : dict modèle → verdict."""
    ms = list(w1)
    gonfle = {NOMS_W1[0], NOMS_W1[1]}
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    if all(w1[m] == NOMS_W1[0] and w2.get(m) == "SYMÉTRIQUE" for m in ms):
        return "SOUTENUE"
    if all(w1[m] in gonfle for m in ms):
        return "PARTIELLE"
    if all(w1[m] == NOMS_W1[2] for m in ms):
        return "RÉFUTÉE"
    if all(w1[m] in gonfle | {NOMS_W1[2]} for m in ms):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


def synthese_simple(v, oui, non, interpretable=True):
    """oui : le verdict qui soutient ; non : les verdicts qui réfutent."""
    vs = list(v.values())
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not vs:
        return "INDÉTERMINÉE"
    if all(x == oui for x in vs):
        return "SOUTENUE"
    if all(x in non for x in vs):
        return "RÉFUTÉE"
    if oui in vs and all(x == oui or x in non for x in vs):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


def apparier(s_ins, h_ins, r_ins, s_pool, h_pool, r_pool, cs=0.5, ch=0.15, nmin=3):
    """Excès d'attention de chaque jeton inséré sur la moyenne des jetons du témoin de même surprise (± cs nats) et de même
    entropie de la loi qui les prédit (± ch) ; nan si moins de nmin voisins."""
    out = np.full(len(s_ins), np.nan)
    for i in range(len(s_ins)):
        m = (np.abs(s_pool - s_ins[i]) <= cs) & (np.abs(h_pool - h_ins[i]) <= ch)
        if m.sum() >= nmin:
            out[i] = r_ins[i] - r_pool[m].mean()
    return out


# Témoins positifs : effets synthétiques par prompt, de taille connue
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
D_pc = 0.05 + rng_pc.normal(0, 0.05, Pp)
E_pc = 0.018 + rng_pc.normal(0, 0.022, Pp)
centre = lambda x: x - x.mean()
bruit = lambda s: centre(rng_pc.normal(0, s, Pp))
M1000 = 0.08 * rng_pc.lognormal(0, 0.5, Pp)
checks = [("W1 gonflé", regle_part(0.2 * D_pc + bruit(0.03), D_pc, bt, NOMS_W1)["verdict"], NOMS_W1[0]),
          ("W1 partiel", regle_part(0.65 * D_pc + bruit(0.03), D_pc, bt, NOMS_W1)["verdict"], NOMS_W1[1]),
          ("W1 conservé", regle_part(D_pc + bruit(0.03), D_pc, bt, NOMS_W1)["verdict"], NOMS_W1[2]),
          ("W1 sans effet du rejeu", regle_part(D_pc, bruit(0.05), bt, NOMS_W1)["verdict"],
           "NON TESTABLE (effet du rejeu non significatif)"),
          ("W2 symétrique", regle_sym(D_pc + bruit(0.03), D_pc, bt)["verdict"], "SYMÉTRIQUE"),
          ("W2 le jeton compte", regle_sym(0.5 * D_pc + bruit(0.03), D_pc, bt)["verdict"], "LE JETON COMPTE"),
          ("W2 inverse plus fort", regle_sym(1.6 * D_pc + bruit(0.03), D_pc, bt)["verdict"], "INVERSE PLUS FORT"),
          ("W3 plateau", regle_v2(M1000 * rng_pc.lognormal(0, 0.1, Pp), M1000, bt)["verdict"], "PLATEAU"),
          ("W3 croissance", regle_v2(1.5 * M1000 * rng_pc.lognormal(0, 0.1, Pp), M1000, bt)["verdict"], "CROISSANCE CONTINUE"),
          ("W3 décroissance", regle_v2(0.6 * M1000 * rng_pc.lognormal(0, 0.1, Pp), M1000, bt)["verdict"], "DÉCROISSANCE"),
          ("W4 lié à la suite", regle_part(0.2 * E_pc + bruit(0.01), E_pc, bt, NOMS_W4)["verdict"], NOMS_W4[0]),
          ("W4 porté par le jeton", regle_part(E_pc + bruit(0.01), E_pc, bt, NOMS_W4)["verdict"], NOMS_W4[2]),
          ("nul équivalent", regle_nul(bruit(0.15), 0.0225, bt)["verdict"], "ÉQUIVALENT À 0"),
          ("nul biaisé", regle_nul(0.05 + rng_pc.normal(0, 0.15, Pp), 0.0225, bt)["verdict"], "BIAISÉ"),
          ("F amplification", regle_F(D_pc, 0.03 + rng_pc.normal(0, 0.05, Pp), bt)["verdict"], "AMPLIFICATION"),
          ("F compensation", regle_F(D_pc, -0.03 + rng_pc.normal(0, 0.05, Pp), bt)["verdict"], "COMPENSATION"),
          ("F équivalence", regle_F(D_pc, bruit(0.02), bt)["verdict"], "ÉQUIVALENT"),
          ("synthèse 1 soutenue", synthese1({"a": NOMS_W1[0], "b": NOMS_W1[0]}, {"a": "SYMÉTRIQUE", "b": "SYMÉTRIQUE"}),
           "SOUTENUE"),
          ("synthèse 1 partielle", synthese1({"a": NOMS_W1[0], "b": NOMS_W1[1]}, {"a": "SYMÉTRIQUE", "b": "SYMÉTRIQUE"}),
           "PARTIELLE"),
          ("synthèse 1 réfutée", synthese1({"a": NOMS_W1[2], "b": NOMS_W1[2]}, {}), "RÉFUTÉE"),
          ("synthèse 1 dépend", synthese1({"a": NOMS_W1[0], "b": NOMS_W1[2]}, {}), "DÉPEND DU MODÈLE"),
          ("synthèse 2 dépend", synthese_simple({"a": NOMS_W4[0], "b": NOMS_W4[2]}, NOMS_W4[0], {NOMS_W4[2]}),
           "DÉPEND DU MODÈLE"),
          ("synthèse 3 réfutée", synthese_simple({"a": "PLATEAU", "b": "DÉCROISSANCE"}, "CROISSANCE CONTINUE",
                                                 {"PLATEAU", "DÉCROISSANCE"}), "RÉFUTÉE")]
s_pool_pc = rng_pc.uniform(0, 10, 20000)
h_pool_pc = rng_pc.uniform(0, 1, 20000)
r_pool_pc = 0.02 + 0.003 * s_pool_pc + rng_pc.normal(0, 0.01, 20000)        # l'attention suit la surprise
s_i, h_i = rng_pc.uniform(2, 8, 300), rng_pc.uniform(0.2, 0.8, 300)
ex0 = apparier(s_i, h_i, 0.02 + 0.003 * s_i + rng_pc.normal(0, 0.01, 300), s_pool_pc, h_pool_pc, r_pool_pc)
ex1 = apparier(s_i, h_i, 0.03 + 0.003 * s_i + rng_pc.normal(0, 0.01, 300), s_pool_pc, h_pool_pc, r_pool_pc)
checks += [("appariement sans excès", "≈ 0" if abs(np.nanmean(ex0)) < 0.002 else f"{np.nanmean(ex0):+.4f}", "≈ 0"),
           ("appariement excès 0,01", "≈ 0,01" if abs(np.nanmean(ex1) - 0.01) < 0.002 else f"{np.nanmean(ex1):+.4f}", "≈ 0,01")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<32} (attendu {att})")
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
    if model_id == "smoke-qwen3-texte":
        # Qwen3 minuscule entraîné par caractères sur le texte des prompts : de vrais effets de contexte (harnais)
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512)
        torch.manual_seed(SEED)
        m, tk = Qwen3ForCausalLM(cfg).to(DEVICE), SmokeTokenizer(512)
        corpus = np.array(tk.encode(" ".join(p for p, *_ in TPL + list(zip(PROMPTS_MAIN, LANG_MAIN))) * 2))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        rng = np.random.default_rng(SEED + 99)
        for _ in range(int(os.environ.get("SMOKE_PAS", "1500"))):
            d = rng.integers(0, len(corpus) - 65, 32)
            x = torch.tensor(np.stack([corpus[i:i + 64] for i in d]), device=DEVICE)
            loss = m(input_ids=x, labels=x).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m.eval(), tk, f"fp32 entraîné sur le texte des prompts (smoke, perte finale {loss.item():.2f})"
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
# ---------- Génération (identique à A1c–A1s ; graine de base du bruit en paramètre) ----------
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
    def sample(self, ctxs, prompt_of, T, nsteps, step0, force=None, base=SEED):
        """Bruit du jeton d'indice step0 + s du prompt p : graine base + 100003 p + step0 + s (base = SEED : celui du
        témoin)."""
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
            seeds = [base + 100003 * prompt_of[j] + step0 + s for j in range(n)]
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


def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque

# %%
# ---------- Témoin (comme A1s), suites libres, passes rejouées ----------
@torch.no_grad()
def generer_temoin(g, enc):
    """Témoin seul (même graine par prompt que A1c–A1s) ; au pas t₀, jeton de chaque rang dans la loi du témoin."""
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
    assert not (jet == tokm[:, T0:T0 + 1]).any(), "un jeton de dose égale celui du témoin"
    print(f"[MESURE] {g.label} génération du témoin : {time.time() - t0:.0f} s", flush=True)
    return tokm, Sg, jet, rg, surp


@torch.no_grad()
def generer_suite(g, enc, tokm, x, base):
    """Suite libre après le jeton x[p] à t₀ (préfixe du témoin) : jetons t₀ + 1 … t₀ + NSUITE ; bruit de graine « base »
    (base = SEED : le bruit du témoin à chaque position, nombres aléatoires communs)."""
    P = len(enc)
    tk, Sg = np.zeros((P, NSUITE), np.int64), np.zeros((P, NSUITE), np.float32)
    for b0 in range(0, P, g.budget):
        rows = list(range(b0, min(P, b0 + g.budget)))
        ctxs = [enc[r] + tokm[r, :T0].tolist() + [int(x[r])] for r in rows]
        S, t = g.sample(ctxs, rows, T_MAIN[0], NSUITE, T0 + 1, base=base)
        tk[rows], Sg[rows] = t, S
    return tk, Sg


@torch.no_grad()
def rejouer(g, ctxs, conts, n_full):
    """Passe rejouée de textes de LONG jetons après le prompt. Pour tous : entropie aux indices t₀ … t₀ + 31 (indice i : loi
    du jeton t₀ + i) et attention de la position qui prédit le jeton t₀ + i vers le jeton t₀ (moyenne sur couches et têtes,
    comme A1s). Pour les n_full premiers (les témoins) : entropie de la loi de chaque jeton j = 0 … LONG, surprise −log p de
    chaque jeton j < LONG, et r₁(j) = attention de j + 1 vers j pour j < LONG − 1 (comme A1v)."""
    base_m, head = g.model.model, g.model.get_output_embeddings()
    n = len(ctxs)
    assert all(len(c) == LONG for c in conts)
    ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    out = base_m(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False, output_attentions=True)
    b = L - LONG                                     # position du jeton 0 de la suite (textes alignés à droite)
    q0, kcol = b - 1 + T0, b + T0                    # requête qui prédit le jeton t₀ ; clé : le jeton t₀
    att = out.attentions
    assert att is not None and att[0] is not None, "poids d'attention indisponibles"
    A = torch.stack([a[:, :, q0:q0 + NPOS, kcol].float().mean(1).to(ids.device) for a in att]).mean(0).cpu().numpy()
    R1 = torch.stack([torch.diagonal(a[:n_full], offset=-1, dim1=-2, dim2=-1)[:, :, b:b + LONG - 1].float().mean(1)
                      .to(ids.device) for a in att]).mean(0).cpu().numpy()
    h = out.last_hidden_state

    def lire(hk, cibles=None):
        e, s = np.zeros(hk.shape[0], np.float64), np.zeros(hk.shape[0], np.float64)
        for c0 in range(0, hk.shape[0], 192):
            z = head(hk[c0:c0 + 192].to(head.weight.device)).float()
            e[c0:c0 + 192] = entropies_from_logits(z, [Q])[Q].cpu().numpy()
            if cibles is not None:
                lp = torch.log_softmax(z.double(), dim=-1)
                s[c0:c0 + 192] = (-lp.gather(1, cibles[c0:c0 + 192].to(lp.device)[:, None])[:, 0]).cpu().numpy()
        return e, s

    S, _ = lire(h[:, q0:q0 + NPOS, :].reshape(n * NPOS, -1))
    cib = torch.cat([ids[:n_full, b:b + LONG], torch.zeros((n_full, 1), dtype=ids.dtype, device=ids.device)], 1)
    Sf, sf = lire(h[:n_full, b - 1:b + LONG, :].reshape(n_full * (LONG + 1), -1), cib.reshape(-1))
    full = {"S": Sf.reshape(n_full, LONG + 1), "surp": sf.reshape(n_full, LONG + 1)[:, :LONG], "r1": R1}
    del out, att, h
    return S.reshape(n, NPOS).astype(np.float32), A.astype(np.float32), full


@torch.no_grad()
def run_replay(g, enc, textes):
    P = len(enc)
    SR = {a: np.zeros((P, NPOS), np.float32) for a in BRAS}
    AR = {a: np.zeros((P, NPOS), np.float32) for a in BRAS}
    F = {"S": np.zeros((P, LONG + 1), np.float32), "surp": np.zeros((P, LONG), np.float32),
         "r1": np.zeros((P, LONG - 1), np.float32)}
    nb = 3 if g.budget >= 48 else 2
    t0 = time.time()
    for b0 in range(0, P, nb):
        rows = list(range(b0, min(P, b0 + nb)))
        m = len(rows)
        S, A, full = rejouer(g, [enc[r] for r in rows] * len(BRAS), [textes[a][r].tolist() for a in BRAS for r in rows], m)
        for ai, a in enumerate(BRAS):
            SR[a][rows], AR[a][rows] = S[ai * m:(ai + 1) * m], A[ai * m:(ai + 1) * m]
        for k in F:
            F[k][rows] = full[k]
        if b0 == 0:
            est = (time.time() - t0) * P / nb / 60
            print(f"[PROGRESSION] {g.label} passes rejouées ({len(BRAS)} textes par prompt) : estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
    print(f"[MESURE] {g.label} passes rejouées : {time.time() - t0:.0f} s", flush=True)
    return SR, AR, F

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
TRANCHES = [(lo, min(hi, MID[1])) for lo, hi in ((2, 7), (8, 15), (16, 31)) if lo <= MID[1]]


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1w.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def fmt(x, c, f="+.5f"):
    return f"{x:{f}} IC95 [{c[0]:{f}} ; {c[1]:{f}}]"


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
        P = len(enc)
        if label in A1S and not SMOKE_TEST:
            a_tok, a_jet = md5(tokm[:, :96]) == A1S[label]["tok"], md5(jet) == A1S[label]["jet"]
            print(f"[ANCRAGE] {label} : témoin identique à A1s (MD5, 768 × 96) : {a_tok} ; jetons des six doses identiques : "
                  f"{a_jet}" + ("" if a_tok and a_jet else " (écart numérique possible ; les contrastes, mesurés ici, restent valides)"))

        # Suites libres, attention par défaut (comme la génération du témoin)
        t1 = time.time()
        own = tokm[:, T0]
        rep, _ = generer_suite(g, enc, tokm, own, SEED)
        ref = tokm[:, T0 + 1:T0 + 1 + NSUITE]
        same, same_tok, same_1 = float(np.mean((rep == ref).all(1))), float(np.mean(rep == ref)), float(np.mean(rep[:, 0] == ref[:, 0]))
        est = (time.time() - t1) * (len(LIBRES) + 1) / 60
        print(f"[VÉRIF] {label} reprise (jeton du témoin, même bruit) : premier jeton identique au témoin pour {same_1:.1%} des "
              f"prompts ; suite entière identique pour {same:.1%} ; jetons identiques {same_tok:.1%} (en fp16, un quasi-ex aequo "
              f"peut basculer et faire diverger la suite sans erreur ; le contrôle décisif est l'accord des entropies plus bas)")
        print(f"[PROGRESSION] {label} suites libres : estimation ~{est:.0f} min", flush=True)
        if same_1 < 0.9 or same < 0.2:
            raise RuntimeError("la génération libre ne redonne pas le témoin : chemin de génération défectueux")
        suites, Sgen = {}, {}
        for i, r in enumerate(RANGS):
            suites[f"lib{r}"], Sgen[f"lib{r}"] = generer_suite(g, enc, tokm, jet[:, i], SEED)
        suites["nul"], Sgen["nul"] = generer_suite(g, enc, tokm, own, FRAIS["nul"])
        for r in R_INV:
            suites[f"libF{r}"], Sgen[f"libF{r}"] = generer_suite(g, enc, tokm, jet[:, RANGS.index(r)], FRAIS[f"libF{r}"])
        print(f"[MESURE] {label} suites libres ({len(LIBRES) + 1} × {NSUITE} jetons) : {time.time() - t1:.0f} s", flush=True)

        textes = {"temoin": tokm[:, :LONG]}
        for i, r in enumerate(RANGS):
            rej = tokm[:, :LONG].copy()
            rej[:, T0] = jet[:, i]
            textes[f"rej{r}"] = rej
            textes[f"lib{r}"] = np.concatenate([tokm[:, :T0], jet[:, i:i + 1], suites[f"lib{r}"]], 1)
        textes["nul"] = np.concatenate([tokm[:, :T0 + 1], suites["nul"]], 1)
        for r in R_INV:
            i = RANGS.index(r)
            textes[f"libF{r}"] = np.concatenate([tokm[:, :T0], jet[:, i:i + 1], suites[f"libF{r}"]], 1)
            textes[f"inv{r}"] = np.concatenate([tokm[:, :T0 + 1], suites[f"libF{r}"]], 1)
        assert set(textes) == set(BRAS) and all(t.shape == (P, LONG) for t in textes.values())
        impl = set_attention(model, "eager")
        print(f"[CONFIG] {label} : passes rejouées en attention « {impl} » (poids d'attention lus)")
        SR, AR, F = run_replay(g, enc, textes)

        dd = np.abs(SR["temoin"] - Sg[:, T0:T0 + NPOS])
        dl = np.concatenate([np.abs(SR[a][:, 1:1 + NSUITE] - Sgen[a]).ravel() for a in LIBRES])
        av = max(float(np.abs(AR[b][:, 0]).max()) for b in BRAS)
        sv = max(float(np.abs(SR[b][:, 0] - SR["temoin"][:, 0]).max()) for b in BRAS)
        fi = max(float(np.abs(F["S"][:, T0:T0 + NPOS] - SR["temoin"]).max()), float(np.abs(F["r1"][:, T0] - AR["temoin"][:, 2]).max()))
        print(f"[VÉRIF] {label} : témoin rejoué contre généré, écart moyen {dd.mean():.1e} (max {dd.max():.1e}) ; suites libres "
              f"rejouées contre générées, écart moyen {dl.mean():.1e} (max {dl.max():.1e}) ; attention vers t₀ avant t₀ (masque "
              f"causal, doit être 0) : max {av:.1e} ; entropie de la loi du jeton t₀ identique entre bras : max {sv:.1e} ; "
              f"lecture complète du témoin contre fenêtre : max {fi:.1e}")
        if dd.mean() > 0.1 or dl.mean() > 0.1 or av > 1e-3 or sv > 0.05 or fi > 1e-2:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")

        w = slice(MID[0], MID[1] + 1)
        moy = lambda a: SR[a][:, w].mean(1).astype(float)
        St = moy("temoin")
        D = {r: moy(f"rej{r}") - St for r in RANGS}
        Tt = {r: moy(f"lib{r}") - St for r in RANGS}
        I = {r: Tt[r] - D[r] for r in RANGS}
        Tnul = moy("nul") - St
        TF = {r: moy(f"libF{r}") - St for r in R_INV}
        Dinv = {r: moy(f"inv{r}") - moy(f"libF{r}") for r in R_INV}
        r1 = {a: AR[a][:, 2].astype(float) for a in BRAS}
        Erej = {r: r1[f"rej{r}"] - r1["temoin"] for r in RANGS}
        Elib = {r: r1[f"lib{r}"] - r1["temoin"] for r in RANGS}
        Arej = {r: (AR[f"rej{r}"] - AR["temoin"])[:, w].mean(1).astype(float) for r in RANGS}
        Alib = {r: (AR[f"lib{r}"] - AR["temoin"])[:, w].mean(1).astype(float) for r in RANGS}
        recouvre = {r: float(np.mean(suites[f"lib{r}"] == ref)) for r in RANGS}
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        cim = lambda x: ci([x[ix].mean() for ix in boots])
        smed = {r: float(np.median(surp[:, i])) for i, r in enumerate(RANGS)}
        a = {"surprise_mediane": smed, "rang_median": {r: float(np.median(rg[:, i])) for i, r in enumerate(RANGS)},
             "reprise": {"premier_jeton": same_1, "prompts": same, "jetons": same_tok}, "recouvrement": recouvre}
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts ; r₁ vers t₀ dans le témoin "
              f"{r1['temoin'].mean():.4f}")
        a["courbes"] = {}
        for r in RANGS:
            cd, ct, cI = cim(D[r]), cim(Tt[r]), cim(I[r])
            a["courbes"][r] = {"D": float(D[r].mean()), "D_ci": cd, "T": float(Tt[r].mean()), "T_ci": ct,
                               "I": float(I[r].mean()), "I_ci": cI}
            print(f"[COURBE] {label} rang {r:>5} (surprise médiane {smed[r]:5.2f} nats) : D {fmt(D[r].mean(), cd)} ; "
                  f"T {fmt(Tt[r].mean(), ct)} ; I {fmt(I[r].mean(), cI)} ; suite libre identique au témoin {recouvre[r]:.0%}")
        if label in A1S and not SMOKE_TEST:
            print(f"[ANCRAGE] {label} D par dose, A1w contre A1s : "
                  + " ; ".join(f"{r} {D[r].mean():.5f}/{A1S[label]['D'][r]:.5f}" for r in RANGS))

        marge = MARGE_REL * abs(float(D[R_HAUT].mean()))
        nul = regle_nul(Tnul, marge, boots, ok)
        t_ok = ok and nul["verdict"] == "ÉQUIVALENT À 0"
        print(f"[TÉMOIN NÉGATIF] {label} T(nul) (jeton du témoin, autre bruit) = {fmt(nul['m'], nul['ci'])} ; marge "
              f"±{marge:.4f} → {nul['verdict']}" + ("" if t_ok or not ok else " : W1 à W4 NON INTERPRÉTABLES"))
        a["temoin_negatif"] = nul
        a["couplage"] = {}
        for r in R_INV:
            dc = TF[r] - Tt[r]
            a["couplage"][r] = {"T_independant": float(TF[r].mean()), "T_couple": float(Tt[r].mean()), "ecart_ci": cim(dc)}
            print(f"[COUPLAGE] {label} rang {r} : T du libre indépendant {TF[r].mean():+.5f} contre T du libre couplé "
                  f"{Tt[r].mean():+.5f} ; écart IC95 [{cim(dc)[0]:+.5f} ; {cim(dc)[1]:+.5f}] (doit contenir 0)")

        w1 = regle_part(Tt[R_W1], D[R_W1], boots, NOMS_W1, t_ok)
        w2 = regle_sym(Dinv[R_W1], D[R_W1], boots, t_ok)
        w3 = regle_v2(Tt[R_SOMMET], Tt[R_HAUT], boots, t_ok)
        w3d = regle_v2(D[R_SOMMET], D[R_HAUT], boots, ok)
        w4 = regle_part(Elib[R_W4], Erej[R_W4], boots, NOMS_W4, t_ok)
        a.update({"W1": w1, "W2": w2, "W3": w3, "W3_D": w3d, "W4": w4})
        print(f"[W1] {label} rang {R_W1} : ρ = T / D = {w1['r']:.2f} IC95 [{w1['ci'][0]:.2f} ; {w1['ci'][1]:.2f}] → {w1['verdict']}")
        print(f"[W2] {label} rang {R_W1} : σ = D_inv / D = {w2['r']:.2f} IC95 [{w2['ci'][0]:.2f} ; {w2['ci'][1]:.2f}] "
              f"(D_inv {Dinv[R_W1].mean():+.5f}) → {w2['verdict']}")
        print(f"[W3] {label} : T({R_SOMMET}) / T({R_HAUT}) = {w3['r']:.2f} IC95 [{w3['ci'][0]:.2f} ; {w3['ci'][1]:.2f}] → "
              f"{w3['verdict']} ; même règle sur D (reproduction d'A1s) : {w3d['r']:.2f} IC95 [{w3d['ci'][0]:.2f} ; "
              f"{w3d['ci'][1]:.2f}] → {w3d['verdict']}")
        print(f"[W4] {label} rang {R_W4} : φ = E_libre / E_rejoué = {w4['r']:.2f} IC95 [{w4['ci'][0]:.2f} ; {w4['ci'][1]:.2f}] "
              f"(E_rejoué {Erej[R_W4].mean():+.5f}, E_libre {Elib[R_W4].mean():+.5f}) → {w4['verdict']}")

        a["F"], a["profil"], a["attention"] = {}, {}, {}
        for r in RANGS:
            f = regle_F(D[r], I[r], boots, t_ok)
            a["F"][r] = f
            print(f"[F] {label} rang {r:>5} : D {fmt(f['D'], f['D_ci'])} ; I {fmt(f['I'], f['I_ci'])} ; I/D {f['I_sur_D']:+.2f} "
                  f"→ {f['verdict']}")
        for r in RANGS:
            pd = [float((SR[f"rej{r}"] - SR["temoin"])[:, lo:hi + 1].mean()) for lo, hi in TRANCHES]
            pt = [float((SR[f"lib{r}"] - SR["temoin"])[:, lo:hi + 1].mean()) for lo, hi in TRANCHES]
            a["profil"][r] = {"D": pd, "T": pt}
            print(f"[PROFIL] {label} rang {r:>5} D / T par tranches de k ({' | '.join(f'{lo}–{hi}' for lo, hi in TRANCHES)}) : "
                  + "  ".join(f"{x:+.4f}/{y:+.4f}" for x, y in zip(pd, pt)))
        for r in R_INV:
            print(f"[INVERSE] {label} rang {r} : D_inv {Dinv[r].mean():+.5f} IC95 [{cim(Dinv[r])[0]:+.5f} ; "
                  f"{cim(Dinv[r])[1]:+.5f}] contre D {D[r].mean():+.5f}")
        a["D_inv"] = {r: {"m": float(Dinv[r].mean()), "ci": cim(Dinv[r])} for r in R_INV}
        for r in RANGS:
            ph = rapport(Elib[r], Erej[r], boots)
            a["attention"][r] = {"E_rejoue": float(Erej[r].mean()), "E_libre": float(Elib[r].mean()), "phi": ph[0],
                                 "phi_ci": ph[1], "A_rejoue": float(Arej[r].mean()), "A_libre": float(Alib[r].mean())}
            print(f"[ATTENTION] {label} rang {r:>5} : excès de r₁ rejoué {Erej[r].mean():+.5f}, libre {Elib[r].mean():+.5f}, "
                  f"φ {ph[0]:.2f} IC95 [{ph[1][0]:.2f} ; {ph[1][1]:.2f}] ; part d'attention vers t₀ sur k = 2–31 (A d'A1s) : "
                  f"rejoué {Arej[r].mean():+.5f}, libre {Alib[r].mean():+.5f}")

        # Comme l'analyse après coup d'A1v : jeton inséré contre jetons du témoin de même surprise et même entropie
        js = np.r_[POOL[0]:T0, T0 + 1:POOL[1] + 1]
        s_pool, h_pool, r_pool = F["surp"][:, js].ravel(), F["S"][:, js].ravel(), F["r1"][:, js].ravel()
        h_t0 = SR["temoin"][:, 0].astype(float)
        a["appariement"] = {}
        cas = [("témoin à t₀ (référence, attendu ≈ 0)", F["surp"][:, T0], r1["temoin"])]
        for r in (RANGS[1], R_W4):
            i = RANGS.index(r)
            cas += [(f"rang {r} rejoué", surp[:, i], r1[f"rej{r}"]), (f"rang {r} libre", surp[:, i], r1[f"lib{r}"])]
        for nom, s_ins, r_ins in cas:
            ex = apparier(s_ins, h_t0, r_ins, s_pool, h_pool, r_pool)
            x = ex[np.isfinite(ex)]
            if len(x) >= 20:
                bx = [x[rngB.integers(0, len(x), len(x))].mean() for _ in range(BOOT_N)]
                a["appariement"][nom] = {"n": int(len(x)), "exces": float(x.mean()), "ci": ci(bx), "part_pos": float(np.mean(x > 0))}
                print(f"[APPARIEMENT] {label} {nom:<38} : n = {len(x):>3} ; excès de r₁ sur les jetons du témoin de même "
                      f"surprise et entropie {fmt(x.mean(), ci(bx))} ; part positive {np.mean(x > 0):.0%}")
            else:
                a["appariement"][nom] = {"n": int(len(x))}
                print(f"[APPARIEMENT] {label} {nom:<38} : effectif insuffisant (n = {len(x)})")
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1w.npz"), tok=tokm, jeton=jet, rang=rg, surprise=surp,
                            **{f"suite_{s}": suites[s] for s in suites}, **{f"S_{b}": SR[b] for b in BRAS},
                            **{f"A_{b}": AR[b] for b in BRAS}, **{f"F_{k}": F[k] for k in F})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1w — l'effet d'un jeton plausible vient-il de la suite rejouée ?")
labs = [lab for lab in ALL if isinstance(ALL[lab], dict) and "W1" in ALL[lab]]
v = {k: {lab: ALL[lab][k]["verdict"] for lab in labs} for k in ("W1", "W2", "W3", "W4")}
for lab in labs:
    print(f"[SYNTHÈSE] {lab:<18} : W1 {v['W1'][lab]:<22} ; W2 {v['W2'][lab]:<18} ; W3 {v['W3'][lab]:<20} ; W4 {v['W4'][lab]}")
manque = len(labs) < len(MODELS)
these1 = "NON TESTABLE (modèle manquant)" if manque else synthese1(v["W1"], v["W2"], ok)
these2 = "NON TESTABLE (modèle manquant)" if manque else synthese_simple(v["W4"], NOMS_W4[0], {NOMS_W4[2]}, ok)
these3 = "NON TESTABLE (modèle manquant)" if manque else synthese_simple(v["W3"], "CROISSANCE CONTINUE",
                                                                          {"PLATEAU", "DÉCROISSANCE"}, ok)
ALL.update({"these_1": these1, "these_2": these2, "these_3": these3})
print(f"[SYNTHÈSE] thèse 1 « dans une passe rejouée, l'effet d'un jeton plausible vient surtout du désaccord avec la suite » : "
      f"{these1} (prédiction SOUTENUE 35 %, PARTIELLE 30 %)")
print(f"[SYNTHÈSE] thèse 2 « l'excès d'attention vers un jeton inséré plausible tient au désaccord avec la suite » : {these2}")
print(f"[SYNTHÈSE] thèse 3 « en génération libre, l'effet croît encore avec la surprise au-delà du rang {R_HAUT} » : {these3}")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

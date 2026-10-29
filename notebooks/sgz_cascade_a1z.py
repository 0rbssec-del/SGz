# %% [markdown]
# # SGZ A1z — L'amplification est-elle une cascade qui passe par les jetons choisis après le choc ?
#
# **Point de départ (A1y, 29 oct.).** Patcher l'état de t₀ fixe aussi la loi du premier jeton après le choc ; A1y a donc
# séparé deux canaux. La loi du premier jeton seule (bruitage à la dernière couche) donne 73 % de l'effet total T, sans
# effet direct ; la trace de t₀ lue par l'attention seule (débruitage à la dernière couche) en donne environ 80 %. Chaque
# canal suffit presque. L'image proposée : l'anomalie aplatit la loi du jeton suivant, le jeton tiré est inhabituel, et
# l'incertitude se propage par les choix suivants (une cascade). A1z teste cette image par des interventions sur les
# jetons eux-mêmes, plus sur les états cachés.
#
# **Dispositif.** Celui d'A1y : 768 prompts, T = 0,6, choc de rang 1000–5000 à t₀ = 64, même graine, même tirage des rangs
# et même bruit de Gumbel que A1c–A1y. Deux modèles : Qwen3-1.7B-Base et SmolLM3-3B-Base. Textes (suite libre = écrite par
# le modèle avec le bruit du témoin, à partir de la première position non forcée) :
# - **témoin** ; **rejoué** (jeton choqué, suite du témoin) ; **choc** (jeton choqué, suite libre) ;
# - **a(n)**, n ∈ {1, 2, 4, 8} : jeton choqué, puis les n premiers jetons **du témoin** forcés, puis suite libre (le choc
#   reste dans le contexte, mais les premiers choix sont ramenés à ceux du témoin) ;
# - **b(n)**, n ∈ {1, 2, 4, 8} : jeton **du témoin** à t₀, puis les n premiers jetons **du choc** forcés, puis suite libre
#   (les choix du choc, sans l'anomalie dans le contexte) ;
# - **b apparié** : jeton du témoin à t₀, puis un premier jeton de même rang (± 1) que celui du choc dans la loi **du
#   témoin**, mais pas choisi en réponse à l'anomalie ; puis suite libre ;
# - **nul** : jeton du témoin, suite libre avec un autre bruit (effet attendu : 0) ;
# - **reprise** : jeton du témoin, même bruit (doit redonner le témoin).
#
# Passes rejouées de tous les textes : entropie de Tsallis (q = 1,45) aux indices t₀ … t₀ + 31.
#
# ## Mesures (par prompt, moyenne sur k = 2–31)
#
# D₀ = S(rejoué) − S(témoin) ; I₀ = S(choc) − S(rejoué) ; T₀ = D₀ + I₀ = S(choc) − S(témoin).
# g(n) = [S(a(n)) − S(rejoué)] / I₀ : part de l'amplification qui subsiste quand les n premiers choix sont ceux du témoin.
# h(n) = [S(b(n)) − S(témoin)] / T₀ : part de l'effet total que reproduisent les n premiers choix du choc, sans l'anomalie.
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# - **Préalable** : I₀ doit être AMPLIFICATION (règle F de A1c) ; sinon Z1 est NON TESTABLE.
# - **Z1 (principal), le premier choix porte l'amplification** : g(1). CASCADE PAR LE PREMIER JETON si la borne haute de
#   l'IC95 < 0,5 ; PARTIEL si IC95 ⊂ [0,5 ; 0,8[ ; PAS PAR LE PREMIER JETON si la borne basse ≥ 0,8 ; sinon INDÉTERMINÉ.
# - **Z2, le choix seul suffit** : h(1). SUFFIT si la borne basse ≥ 0,5 ; PARTIEL si IC95 ⊂ [0,25 ; 0,5[ ; NE SUFFIT PAS
#   si la borne haute < 0,25 ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de T₀ contient 0.
# - **Z3, est-ce la surprise du choix ou son lien avec l'anomalie ?** ρ = T(b(1)) / T(b apparié). SURPRISE DU CHOIX SEULE si
#   IC95 ⊂ [0,8 ; 1,25] ; LIEN AVEC L'ANOMALIE si la borne basse > 1,25 ; CHOIX APPARIÉ PLUS FORT si la borne haute < 0,8 ;
#   sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de T(b apparié) contient 0.
# - **Témoin négatif** : T(nul) équivalent à 0, IC95 ⊂ [−0,3 × D₀ ; +0,3 × D₀] ; sinon Z1 à Z3 NON INTERPRÉTABLES.
# - **Synthèse, thèse « l'amplification est une cascade qui passe par le premier jeton choisi après le choc »** : SOUTENUE
#   si Z1 = CASCADE PAR LE PREMIER JETON et Z2 = SUFFIT pour les deux modèles ; RÉFUTÉE si Z1 = PAS PAR LE PREMIER JETON et
#   Z2 = NE SUFFIT PAS pour les deux ; DÉPEND DU MODÈLE si un modèle suit le premier schéma et l'autre le second ; sinon
#   INDÉTERMINÉE.
# - **Descriptifs** : courbes g(n) et h(n) avec IC95 ; part des prompts où le premier jeton du choc est celui du témoin ;
#   surprise (dans la loi du témoin) du premier jeton du choc et du jeton apparié ; Spearman par prompt entre cette
#   surprise et T(b(1)) ; profils par tranches de k.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Z1 : CASCADE 40 %, PARTIEL 25 %, INDÉTERMINÉ 25 %. Z2 : SUFFIT 55 %, PARTIEL 25 %. Z3 : SURPRISE DU CHOIX SEULE 40 %,
# LIEN AVEC L'ANOMALIE 25 %, INDÉTERMINÉ 35 %. Synthèse SOUTENUE 30 %, RÉFUTÉE 10 %. g(n) décroissant avec n : 80 %.
#
# ## Limites connues
#
# - Dans b(n), les jetons transplantés ont été choisis après l'anomalie et sont placés derrière le jeton du témoin : leur
#   désaccord avec le contexte fait partie de ce qui est mesuré (comme l'inverse d'A1w). Le bras apparié contrôle leur
#   surprise, pas leur sens.
# - g et h sont des parts d'un même I₀ ou T₀ : leur niveau hérite de l'incertitude de ce dénominateur.
# - Une bande de chocs (rang 1000–5000), une température, deux modèles ; anomalies artificielles.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~20 min pour Qwen3,
# ~30 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[TÉMOIN NÉGATIF]`, `[F]`,
# `[Z1]` à `[Z3]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1z" if os.path.isdir("/kaggle/working") else "sgz_a1z_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T1 = 768, 96, 64
RANK_LO, RANK_HI = 1000, 5000
MID = (2, 31)
MARGE_REL = 0.3
NS = [1, 2, 4, 8]                                # nombres de jetons forcés après t₀
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1y
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1Y = {"Qwen3-1.7B-Base": {"D": 0.07734, "I": 0.03958}}                       # sans crochet, 768 prompts
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1 = int(os.environ.get("SMOKE_N", "16")), 48, 12
    RANK_LO, RANK_HI = 20, 80
    MID = (2, 12)
    NS = [1, 2, 4]
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
NSUITE = NPOS - 2                                # jetons après t₀ : t₀ + 1 … t₀ + 30
LONG = T1 + 1 + NSUITE                           # textes rejoués : jetons 0 … t₀ + 30
FRAIS = SEED + 7919                              # autre bruit (bras nul)
assert LONG <= GEN_LEN and max(NS) < NSUITE
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T1} fenêtre k={MID[0]}–{MID[1]} ; jetons forcés n ∈ {NS} ; "
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


def rapport(x, y, boots):
    """Rapport des moyennes x̄ / ȳ, IC95 du rapport et IC95 de ȳ (mêmes rééchantillons)."""
    return float(x.mean() / y.mean()), ci([x[ix].mean() / y[ix].mean() for ix in boots]), ci([y[ix].mean() for ix in boots])


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
    return {"D": d, "D_ci": dci, "I": i, "I_ci": ici, "verdict": v}


def regle_z1(Ia, I0, boots, amp, interpretable=True):
    """g(1) = Ī(a(1)) / Ī₀."""
    r, c, _ = rapport(Ia, I0, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif not amp:
        v = "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"
    elif c[1] < 0.5:
        v = "CASCADE PAR LE PREMIER JETON"
    elif c[0] >= 0.5 and c[1] < 0.8:
        v = "PARTIEL"
    elif c[0] >= 0.8:
        v = "PAS PAR LE PREMIER JETON"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def regle_z2(Tb, T0, boots, interpretable=True):
    """h(1) = T̄(b(1)) / T̄₀."""
    r, c, cy = rapport(Tb, T0, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (T₀ non significatif)"
    elif c[0] >= 0.5:
        v = "SUFFIT"
    elif c[0] >= 0.25 and c[1] < 0.5:
        v = "PARTIEL"
    elif c[1] < 0.25:
        v = "NE SUFFIT PAS"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def regle_z3(Tb, Tapp, boots, interpretable=True):
    """ρ = T̄(b(1)) / T̄(b apparié)."""
    r, c, cy = rapport(Tb, Tapp, boots)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (bras apparié sans effet)"
    elif c[0] >= 0.8 and c[1] <= 1.25:
        v = "SURPRISE DU CHOIX SEULE"
    elif c[0] > 1.25:
        v = "LIEN AVEC L'ANOMALIE"
    elif c[1] < 0.8:
        v = "CHOIX APPARIÉ PLUS FORT"
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


def synthese(z1, z2, interpretable=True):
    """z1, z2 : dict modèle → verdict."""
    ms = list(z1)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    oui = [m for m in ms if z1[m] == "CASCADE PAR LE PREMIER JETON" and z2.get(m) == "SUFFIT"]
    non = [m for m in ms if z1[m] == "PAS PAR LE PREMIER JETON" and z2.get(m) == "NE SUFFIT PAS"]
    if len(oui) == len(ms):
        return "SOUTENUE"
    if len(non) == len(ms):
        return "RÉFUTÉE"
    if oui and non and len(oui) + len(non) == len(ms):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : effets synthétiques par prompt, de taille connue
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
I0_pc = 0.04 + rng_pc.normal(0, 0.15, Pp)
T0_pc = 0.12 + rng_pc.normal(0, 0.15, Pp)
bruit = lambda: (lambda x: x - x.mean())(rng_pc.normal(0, 0.03, Pp))
checks = [("Z1 cascade", regle_z1(0.1 * I0_pc + bruit(), I0_pc, bt, True)["verdict"], "CASCADE PAR LE PREMIER JETON"),
          ("Z1 partiel", regle_z1(0.65 * I0_pc + bruit(), I0_pc, bt, True)["verdict"], "PARTIEL"),
          ("Z1 pas par le premier", regle_z1(I0_pc + bruit(), I0_pc, bt, True)["verdict"], "PAS PAR LE PREMIER JETON"),
          ("Z1 sans amplification", regle_z1(0.1 * I0_pc, I0_pc, bt, False)["verdict"], "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"),
          ("Z2 suffit", regle_z2(0.8 * T0_pc + bruit(), T0_pc, bt)["verdict"], "SUFFIT"),
          ("Z2 partiel", regle_z2(0.37 * T0_pc + bruit(), T0_pc, bt)["verdict"], "PARTIEL"),
          ("Z2 ne suffit pas", regle_z2(0.05 * T0_pc + bruit(), T0_pc, bt)["verdict"], "NE SUFFIT PAS"),
          ("Z3 surprise seule", regle_z3(T0_pc + bruit(), T0_pc, bt)["verdict"], "SURPRISE DU CHOIX SEULE"),
          ("Z3 lien", regle_z3(1.8 * T0_pc + bruit(), T0_pc, bt)["verdict"], "LIEN AVEC L'ANOMALIE"),
          ("Z3 apparié plus fort", regle_z3(0.5 * T0_pc + bruit(), T0_pc, bt)["verdict"], "CHOIX APPARIÉ PLUS FORT"),
          ("nul équivalent", regle_nul(bruit(), 0.02, bt)["verdict"], "ÉQUIVALENT À 0"),
          ("nul biaisé", regle_nul(0.05 + rng_pc.normal(0, 0.15, Pp), 0.02, bt)["verdict"], "BIAISÉ"),
          ("synthèse soutenue", synthese({"a": "CASCADE PAR LE PREMIER JETON", "b": "CASCADE PAR LE PREMIER JETON"},
                                         {"a": "SUFFIT", "b": "SUFFIT"}), "SOUTENUE"),
          ("synthèse réfutée", synthese({"a": "PAS PAR LE PREMIER JETON", "b": "PAS PAR LE PREMIER JETON"},
                                        {"a": "NE SUFFIT PAS", "b": "NE SUFFIT PAS"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "CASCADE PAR LE PREMIER JETON", "b": "PAS PAR LE PREMIER JETON"},
                                       {"a": "SUFFIT", "b": "NE SUFFIT PAS"}), "DÉPEND DU MODÈLE"),
          ("synthèse indéterminée", synthese({"a": "PARTIEL", "b": "CASCADE PAR LE PREMIER JETON"},
                                             {"a": "SUFFIT", "b": "SUFFIT"}), "INDÉTERMINÉE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<42} (attendu {att})")
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
# ---------- Génération (identique à A1c–A1y ; graine de base du bruit en paramètre) ----------
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
        self.patch = None                        # crochet de patching de la position t₀

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
        n = len(ctxs)
        S, tk = np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int64)
        ids, attn, _ = self.left_pad(ctxs)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        past = None
        for s in range(nsteps):
            if self.patch is not None and self.patch.val_gen is not None:   # t₀ = dernière position du préremplissage
                self.patch.p, self.patch.val = (ids.shape[1] - 1, self.patch.val_gen) if s == 0 else (None, None)
            out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
            e1 = entropies_from_logits(z, [Q])[Q].cpu().numpy()
            seeds = [base + 100003 * prompt_of[j] + step0 + s for j in range(n)]   # base = SEED : bruit du témoin
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
        if self.patch is not None:
            self.patch.p, self.patch.val = None, None
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
# ---------- Choc à t₀ (comme A1c–A1y), suites libres et passes rejouées ----------
def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque


def sortie(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def run_main(g, enc, idx, X_fixe=None):
    """Témoin et choc empilés pour les prompts idx (indices d'origine : mêmes graines que A1c–A1t). Sans X_fixe : X au rang
    tiré à t₀, jamais le jeton du témoin ; avec X_fixe : le même jeton que dans la condition de référence."""
    forced_rank = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, N_PROMPTS)
    P = len(idx)
    res = {a: {"S": np.zeros((P, GEN_LEN), np.float32), "tok": np.zeros((P, GEN_LEN), np.int64)} for a in ("temoin", "choc")}
    info = {"X": np.zeros(P, np.int64), "rang": np.zeros(P, np.int64), "surp": np.zeros(P), "egal": np.zeros(P, bool)}
    npb = max(1, g.budget // 2)
    for b0 in range(0, P, npb):
        loc = list(range(b0, min(P, b0 + npb)))
        rows_p = [int(idx[i]) for i in loc]
        n = len(loc)

        def force(s, z, nxt, n=n, loc=loc, rows_p=rows_p):
            if s < T1:
                nxt[n:] = nxt[:n]                                      # appariement exact avant le choc
            elif s == T1:
                if X_fixe is None:
                    zz = g.masque(z[n:].clone())
                    lp = torch.log_softmax(z[n:].double(), dim=-1)
                    order = torch.argsort(zz, dim=-1, descending=True)
                for j, (i, r) in enumerate(zip(loc, rows_p)):
                    if X_fixe is None:
                        rk = int(min(forced_rank[r], order.shape[1] - 2))
                        if int(order[j, rk]) == int(nxt[j]):         # jamais le jeton du témoin
                            rk += 1
                        x = int(order[j, rk])
                        info["rang"][i], info["surp"][i] = rk, -float(lp[j, x])
                    else:
                        x = int(X_fixe[i])
                    info["X"][i], info["egal"][i] = x, x == int(nxt[j])
                    nxt[n + j] = x
            return nxt

        S, tk = g.sample([enc[r] for r in rows_p] * 2, rows_p * 2, T_MAIN[0], GEN_LEN, 0, force)
        for ai, a in enumerate(("temoin", "choc")):
            sl = slice(ai * n, (ai + 1) * n)
            res[a]["S"][loc], res[a]["tok"][loc] = S[sl], tk[sl]
    ref, fo = res["temoin"]["tok"], res["choc"]["tok"]
    if not np.array_equal(ref[:, :T1], fo[:, :T1]) or float(np.mean(ref[:, T1] != fo[:, T1])) < 0.95:
        raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
    return res, info



@torch.no_grad()
def generer_depuis(g, enc, idx, pref, base=SEED):
    """Suite libre après prompt + pref (P, p) : jetons p … LONG − 1, bruit de graine « base » (SEED : le bruit du témoin à
    chaque position, nombres aléatoires communs). Renvoie le texte complet (P, LONG)."""
    P, p = pref.shape
    tk = np.zeros((P, LONG - p), np.int64)
    for b0 in range(0, P, g.budget):
        loc = list(range(b0, min(P, b0 + g.budget)))
        _, t = g.sample([enc[int(idx[i])] + pref[i].tolist() for i in loc], [int(idx[i]) for i in loc], T_MAIN[0], LONG - p, p,
                        base=base)
        tk[loc] = t
    return np.concatenate([pref, tk], 1)


@torch.no_grad()
def rejouer_S(g, enc, idx, textes):
    """Entropie aux indices t₀ … t₀ + 31 de textes de LONG jetons (indice i : loi du jeton t₀ + i)."""
    base, head = g.model.model, g.model.get_output_embeddings()
    P = len(idx)
    S = np.zeros((P, NPOS), np.float32)
    nb = 24
    for b0 in range(0, P, nb):
        loc = list(range(b0, min(P, b0 + nb)))
        ids, attn, L = g.left_pad([enc[int(idx[i])] + textes[i].tolist() for i in loc])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
        b = L - LONG                                 # position du jeton 0 de la suite (textes alignés à droite)
        hk = h[:, b + T1 - 1:b + T1 - 1 + NPOS, :].reshape(len(loc) * NPOS, -1)
        flat = np.zeros(len(loc) * NPOS)
        for c0 in range(0, len(flat), 192):
            flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
        S[loc] = flat.reshape(len(loc), NPOS)
    return S


@torch.no_grad()
def jeton_apparie(g, enc, idx, tokm, c):
    """Loi du témoin au pas t₀ + 1 (prompt + jetons 0 … t₀ du témoin). Pour le premier jeton du choc c : son rang et sa
    surprise −log p dans cette loi, et le jeton de rang voisin (rang + 1, en sautant le jeton du témoin) avec sa surprise.
    Si c est le jeton du témoin, le jeton apparié l'est aussi (aucun changement)."""
    P = len(idx)
    app, rk = np.zeros(P, np.int64), np.zeros(P, np.int64)
    s_c, s_a = np.zeros(P), np.zeros(P)
    own = tokm[:, T1 + 1]
    for b0 in range(0, P, g.budget):
        loc = list(range(b0, min(P, b0 + g.budget)))
        ids, attn, _ = g.left_pad([enc[int(idx[i])] + tokm[i, :T1 + 1].tolist() for i in loc])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        z = g.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).logits[:, -1, :].float()
        lp = torch.log_softmax(z.double(), dim=-1)
        order = torch.argsort(g.masque(z.clone()), dim=-1, descending=True)
        for j, i in enumerate(loc):
            ci_ = int(c[i])
            r = int((order[j] == ci_).nonzero()[0, 0])
            rk[i], s_c[i] = r, -float(lp[j, ci_])
            if ci_ == int(own[i]):
                app[i], s_a[i] = ci_, s_c[i]
                continue
            k = r + 1
            while int(order[j, k]) == int(own[i]):
                k += 1
            app[i], s_a[i] = int(order[j, k]), -float(lp[j, int(order[j, k])])
    return app, s_c, s_a, rk

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
W = slice(MID[0], MID[1] + 1)
TRANCHES = [(lo, min(hi, MID[1])) for lo, hi in ((2, 7), (8, 15), (16, 31)) if lo <= MID[1]]


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1z.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def fmt(x, c, f="+.4f"):
    return f"{x:{f}} IC95 [{c[0]:{f}} ; {c[1]:{f}}]"


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
        enc = [tok.encode(p)[:12] if SMOKE_TEST else tok.encode(p) for p in PROMPTS]     # comme A1k
        g = Gen(model, tok, label)
        idx = np.arange(len(enc))
        P = len(idx)
        t0g = time.time()
        res, info = run_main(g, enc, idx)
        print(f"[MESURE] {label} génération témoin + choc : {time.time() - t0g:.0f} s ; rang médian du choc "
              f"{np.median(info['rang']):.0f}, surprise médiane {np.median(info['surp']):.2f} nats", flush=True)
        tokm, X = res["temoin"]["tok"], info["X"]
        pre = tokm[:, :T1]
        tg = time.time()
        rep = generer_depuis(g, enc, idx, tokm[:, :T1 + 1])
        same_1 = float(np.mean(rep[:, T1 + 1] == tokm[:, T1 + 1]))
        same = float(np.mean((rep[:, T1 + 1:LONG] == tokm[:, T1 + 1:LONG]).all(1)))
        n_gen = 3 + 2 * len(NS) + 1
        print(f"[VÉRIF] {label} reprise (jeton du témoin, même bruit) : premier jeton identique au témoin {same_1:.1%} ; suite "
              f"entière identique {same:.1%} (en fp16, un quasi-ex aequo peut basculer sans erreur)")
        print(f"[PROGRESSION] {label} suites libres : estimation ~{(time.time() - tg) * n_gen / 60:.0f} min pour {n_gen} suites",
              flush=True)
        if same_1 < 0.9 or same < 0.2:
            raise RuntimeError("la génération libre ne redonne pas le témoin : chemin de génération défectueux")
        textes = {"temoin": tokm[:, :LONG].copy()}
        textes["rejoue"] = textes["temoin"].copy()
        textes["rejoue"][:, T1] = X
        textes["choc"] = generer_depuis(g, enc, idx, np.concatenate([pre, X[:, None]], 1))
        same_c = float(np.mean((textes["choc"][:, T1 + 1:LONG] == res["choc"]["tok"][:, T1 + 1:LONG]).all(1)))
        c1 = textes["choc"][:, T1 + 1]
        for n in NS:
            textes[f"a{n}"] = generer_depuis(g, enc, idx, np.concatenate([pre, X[:, None], tokm[:, T1 + 1:T1 + 1 + n]], 1))
            textes[f"b{n}"] = generer_depuis(g, enc, idx, np.concatenate([tokm[:, :T1 + 1], textes["choc"][:, T1 + 1:T1 + 1 + n]], 1))
        app, s_c, s_a, rk_c = jeton_apparie(g, enc, idx, tokm, c1)
        textes["bapp"] = generer_depuis(g, enc, idx, np.concatenate([tokm[:, :T1 + 1], app[:, None]], 1))
        textes["nul"] = generer_depuis(g, enc, idx, tokm[:, :T1 + 1], base=FRAIS)
        print(f"[MESURE] {label} suites libres : {time.time() - tg:.0f} s", flush=True)
        for n in NS:
            assert (textes[f"a{n}"][:, T1] == X).all() and (textes[f"a{n}"][:, T1 + 1:T1 + 1 + n] == tokm[:, T1 + 1:T1 + 1 + n]).all()
            assert (textes[f"b{n}"][:, :T1 + 1] == tokm[:, :T1 + 1]).all()
        tr = time.time()
        SR = {k: rejouer_S(g, enc, idx, v) for k, v in textes.items()}
        print(f"[MESURE] {label} passes rejouées ({len(textes)} textes) : {time.time() - tr:.0f} s", flush=True)
        dd = np.abs(SR["temoin"] - res["temoin"]["S"][:, T1:T1 + NPOS])
        sv = max(float(np.abs(SR[k][:, 0] - SR["temoin"][:, 0]).max()) for k in SR)
        print(f"[VÉRIF] {label} : témoin rejoué contre généré, écart moyen {dd.mean():.1e} (max {dd.max():.1e}) ; entropie de la "
              f"loi du jeton t₀ identique entre textes : max {sv:.1e} ; suite libre du choc régénérée identique à la génération "
              f"complète : {same_c:.1%} ; premier jeton du choc égal à celui du témoin : {np.mean(c1 == tokm[:, T1 + 1]):.1%}")
        if dd.mean() > 0.1 or sv > 0.05:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")

        moy = lambda k: SR[k][:, W].mean(1).astype(float)
        St, Sr, Sc = moy("temoin"), moy("rejoue"), moy("choc")
        D0, I0, T0 = Sr - St, Sc - Sr, Sc - St
        Ia = {n: moy(f"a{n}") - Sr for n in NS}
        Tb = {n: moy(f"b{n}") - St for n in NS}
        Tapp, Tnul = moy("bapp") - St, moy("nul") - St
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        cim = lambda x: ci([x[ix].mean() for ix in boots])
        F0 = regle_F(D0, I0, boots, ok)
        amp = F0["verdict"] == "AMPLIFICATION"
        nul = regle_nul(Tnul, MARGE_REL * abs(float(D0.mean())), boots, ok)
        t_ok = ok and nul["verdict"] == "ÉQUIVALENT À 0"
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts")
        print(f"[F] {label} : D₀ = {fmt(F0['D'], F0['D_ci'])} ; I₀ = {fmt(F0['I'], F0['I_ci'])} → {F0['verdict']} ; T₀ = "
              f"{fmt(T0.mean(), cim(T0))}" + (f" (A1y : D {A1Y[label]['D']:+.4f}, I {A1Y[label]['I']:+.4f})" if label in A1Y else ""))
        print(f"[TÉMOIN NÉGATIF] {label} T(nul) (jeton du témoin, autre bruit) = {fmt(nul['m'], nul['ci'])} ; marge "
              f"±{nul['marge']:.4f} → {nul['verdict']}" + ("" if t_ok or not ok else " : Z1 à Z3 NON INTERPRÉTABLES"))
        a = {"F0": F0, "T0": float(T0.mean()), "T0_ci": cim(T0), "temoin_negatif": nul, "reprise": {"premier": same_1, "suite": same},
             "choc_regenere_identique": same_c, "premier_jeton_inchange": float(np.mean(c1 == tokm[:, T1 + 1])), "g": {}, "h": {}}
        for n in NS:
            gr = rapport(Ia[n], I0, boots)
            hr = rapport(Tb[n], T0, boots)
            a["g"][n] = {"r": gr[0], "ci": gr[1], "I": float(Ia[n].mean())}
            a["h"][n] = {"r": hr[0], "ci": hr[1], "T": float(Tb[n].mean())}
            print(f"[COURBE] {label} n = {n} : g(n) = I(a(n)) / I₀ = {gr[0]:+.2f} IC95 [{gr[1][0]:+.2f} ; {gr[1][1]:+.2f}] "
                  f"(I {Ia[n].mean():+.4f}) ; h(n) = T(b(n)) / T₀ = {hr[0]:+.2f} IC95 [{hr[1][0]:+.2f} ; {hr[1][1]:+.2f}] "
                  f"(T {Tb[n].mean():+.4f})")
        z1 = regle_z1(Ia[1], I0, boots, amp, t_ok)
        z2 = regle_z2(Tb[1], T0, boots, t_ok)
        z3 = regle_z3(Tb[1], Tapp, boots, t_ok)
        a.update({"Z1": z1, "Z2": z2, "Z3": z3, "T_apparie": float(Tapp.mean()), "T_apparie_ci": cim(Tapp)})
        print(f"[Z1] {label} premier choix ramené à celui du témoin : g(1) = {z1['r']:+.2f} IC95 [{z1['ci'][0]:+.2f} ; "
              f"{z1['ci'][1]:+.2f}] → {z1['verdict']}")
        print(f"[Z2] {label} premier choix du choc sans l'anomalie : h(1) = {z2['r']:+.2f} IC95 [{z2['ci'][0]:+.2f} ; "
              f"{z2['ci'][1]:+.2f}] → {z2['verdict']}")
        print(f"[Z3] {label} choix du choc contre choix apparié : T(b(1)) / T(apparié) = {z3['r']:.2f} IC95 [{z3['ci'][0]:.2f} ; "
              f"{z3['ci'][1]:.2f}] (T apparié {Tapp.mean():+.4f}) → {z3['verdict']}")
        chg = c1 != tokm[:, T1 + 1]
        rho = float(np.corrcoef(np.argsort(np.argsort(s_c[chg])), np.argsort(np.argsort(Tb[1][chg])))[0, 1]) if chg.sum() > 10 else float("nan")
        a["surprise"] = {"choc_median": float(np.median(s_c[chg])) if chg.any() else float("nan"),
                         "apparie_median": float(np.median(s_a[chg])) if chg.any() else float("nan"),
                         "rang_median": float(np.median(rk_c[chg])) if chg.any() else float("nan"), "spearman_Tb1": rho}
        print(f"[SURPRISE] {label} premier jeton du choc dans la loi du témoin (prompts où il diffère, {chg.mean():.0%}) : rang médian "
              f"{a['surprise']['rang_median']:.0f}, surprise médiane {a['surprise']['choc_median']:.2f} nats ; jeton apparié "
              f"{a['surprise']['apparie_median']:.2f} nats ; Spearman(surprise, T(b(1))) {rho:+.3f}")
        a["profil"] = {}
        for k in ["choc", "rejoue", "a1", "b1", "bapp"]:
            pr = [float((SR[k] - SR["temoin"])[:, lo:hi + 1].mean()) for lo, hi in TRANCHES]
            a["profil"][k] = pr
            print(f"[PROFIL] {label} {k:<6} − témoin par tranches de k ({' | '.join(f'{lo}–{hi}' for lo, hi in TRANCHES)}) : "
                  + "  ".join(f"{x:+.4f}" for x in pr))
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1z.npz"), surp_choc=s_c, surp_app=s_a, rang_choc=rk_c,
                            **{f"S_{k}": v for k, v in SR.items()}, **{f"tok_{k}": v for k, v in textes.items()})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1z — l'amplification est-elle une cascade qui passe par les jetons choisis après le choc ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "Z1" in ALL[m]]
v = {k: {m: ALL[m][k]["verdict"] for m in labs} for k in ("Z1", "Z2", "Z3")}
for m in labs:
    print(f"[SYNTHÈSE] {m:<18} : Z1 {v['Z1'][m]:<30} ; Z2 {v['Z2'][m]:<14} ; Z3 {v['Z3'][m]}")
these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese(v["Z1"], v["Z2"], ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « l'amplification est une cascade qui passe par le premier jeton choisi après le choc » : {these} "
      f"(prédiction SOUTENUE 30 %, RÉFUTÉE 10 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

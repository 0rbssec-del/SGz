# %% [markdown]
# # SGZ A1r — La complexité d'une surprise s'additionne-t-elle à son intensité, ou la multiplie-t-elle ?
#
# **Thèse d'Erick (28 oct. 2026).** « La complexité d'une surprise augmente l'intensité de la surprise, donc de
# l'attention soudaine », peut-être de façon multiplicative. **Version testée** : une anomalie qui viole deux niveaux à la
# fois (le **sens** : jeton improbable en contexte ; la **forme** : jeton d'une autre écriture que le texte) attire plus
# d'attention qu'une anomalie de même surprise qui ne viole que le sens, et ce surplus **grandit avec la surprise**
# (multiplicatif) au lieu de rester constant (additif).
#
# **Ce qui existe déjà.** A2 (même projet) : cinq types de corruption combinés font 0,82 [0,73 ; 0,92] fois la somme de
# leurs effets séparés sur l'entropie (règle : ADDITIF ; sens : sous-additif). A1p : dans la bande des jetons très rares,
# 3 nats de surprise en plus ne changent presque pas l'effet direct (plafond possible, après coup). En lecture humaine, le
# temps de lecture croît à peu près linéairement avec la surprise (Smith & Levy, 2013 ; cité de mémoire, non vérifié ici).
# Aucun test de l'interaction forme × sens sur l'attention d'un LLM trouvé [non cherché de façon systématique].
#
# **Dispositif.** Témoin seul généré (768 prompts, T = 0,6, 96 jetons, comme A1k), puis cinq passes rejouées (forçage
# par l'enseignant) où seul le jeton t₀ = 64 change, la suite restant celle du témoin. Plan 2 × 2 à surprise appariée :
#
# | Bras | Écriture du jeton t₀ | Surprise dans le contexte | Choix |
# |---|---|---|---|
# | L_mod | latine (un mot) | modérée | rang voisin de NL_mod |
# | NL_mod | non latine | modérée | le jeton non latin le plus probable au-delà du rang 20 (rang ≤ 500) |
# | L_fort | latine (un mot) | forte | rang voisin de NL_fort |
# | NL_fort | non latine | forte | le jeton non latin le plus proche d'un rang tiré entre 1000 et 5000 |
#
# Écriture lue sur le jeton décodé : « latin » = un mot de 2 lettres latines ou plus (espace initiale permise) ;
# « non latin » = au moins une lettre, toutes hors de l'alphabet latin (CJK, cyrillique, arabe…), aucun octet incomplet.
# Ω = prompts où les quatre jetons existent.
#
# ## Mesures
#
# - **A (principale, l'« attention soudaine »)** : part d'attention vers la position t₀, moyenne sur les couches, les têtes
#   et les positions k = 2–31 après t₀, moins la même part dans le témoin.
# - **D (secondaire)** : effet direct sur l'entropie de Tsallis (q = 1,45), moyenne de bras − témoin sur k = 2–31.
#
# Pour chaque mesure, a, b, c, d = moyennes de L_mod, NL_mod, L_fort, NL_fort sur Ω ; bootstrap apparié sur les prompts
# (B = 1000).
#
# ## Critères (fixés avant toute exécution)
#
# - **Préalables** (sinon NON TESTABLE) : |Ω| ≥ 200 ; appariement : écart médian de surprise |NL − L| ≤ 0,25 nat à chaque
#   niveau ; niveaux distincts : surprise médiane forte − modérée ≥ 3 nats.
# - **V1, la forme amplifie-t-elle à surprise égale (niveau fort) ?** F = d / c : AMPLIFIE si IC95 > 1 ; ATTÉNUE si
#   IC95 < 1 ; ÉQUIVALENT si IC95 ⊂ [0,8 ; 1,2] ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de c contient 0.
#   Même règle, descriptive, au niveau modéré (b / a).
# - **V2, comment la forme se combine-t-elle à la surprise ?** Interaction additive Int = (d − c) − (b − a) ; interaction
#   multiplicative Lr = ln(d / c) − ln(b / a). Additif : Int = 0 ; multiplicatif : Lr = 0 (donc Int > 0).
#   NON TESTABLE si un préalable manque, si l'effet de la surprise n'est pas positif (IC95 de c − a ⊄ ]0 ; +∞[ : les deux
#   modèles ne se distinguent plus), ou si l'effet de forme n'est pas positif aux deux niveaux (b − a ou d − c ≤ 0, ou
#   aucun des deux IC95 au-dessus de 0). Sinon, dans l'ordre : SUPRA-MULTIPLICATIF si IC95(Lr) > 0 ; si IC95(Int) > 0 :
#   MULTIPLICATIF si IC95(Lr) contient 0, sinon SUR-ADDITIF (entre l'additif et le multiplicatif) ; SOUS-ADDITIF si
#   IC95(Int) < 0 ; ADDITIF si IC95(Lr) < 0 ; sinon NON DISCRIMINANT.
# - **Synthèse sur A, thèse « la complexité multiplie l'intensité de la surprise »** : SOUTENUE si V2(A) ∈
#   {MULTIPLICATIF, SUPRA-MULTIPLICATIF} pour les deux modèles ; PARTIELLE si V2(A) ∈ {SUR-ADDITIF, MULTIPLICATIF,
#   SUPRA-MULTIPLICATIF} pour les deux sans être multiplicatif deux fois ; RÉFUTÉE si V2(A) ∈ {ADDITIF, SOUS-ADDITIF}
#   pour les deux ; DÉPEND DU MODÈLE si l'un est au-dessus de l'additif et l'autre additif ou sous-additif ; sinon
#   INDÉTERMINÉE. Même synthèse, secondaire, sur D.
# - **Descriptifs** : surprise et rang médians par bras, écritures des jetons non latins, profil selon k, valeurs à k = 1.
# - Témoins positifs synthétiques pour chaque règle (additif, multiplicatif, supra, sur-additif, sous-additif, sans effet
#   de forme, sans effet de surprise).
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1(A) AMPLIFIE 45 %, ÉQUIVALENT 30 %, ATTÉNUE 15 %. V2(A) : SOUS-ADDITIF 30 %, ADDITIF 20 %, NON DISCRIMINANT ou NON
# TESTABLE 25 %, SUR-ADDITIF 10 %, MULTIPLICATIF ou SUPRA 15 %. Synthèse sur A : SOUTENUE 10 %, PARTIELLE 10 %,
# RÉFUTÉE 40 %. Raisons : sous-additivité dans A2, plafond possible dans A1p.
#
# ## Limites connues
#
# - « Complexité » est réduite à deux niveaux (écriture, contexte) ; un jeton non latin probable peut être une traduction
#   correcte (violation de forme sans violation de sens) : le niveau modéré mélange donc les deux cas.
# - Surprise appariée par le rang voisin, pas par le sens ; la fréquence d'usage des jetons non latins n'est pas appariée.
# - Effet direct et attention mesurés par passes rejouées (la suite reste celle du témoin).
# - Deux modèles de familles différentes (Qwen3-1.7B-Base, SmolLM3-3B-Base), une température.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~15 min pour Qwen3,
# ~25 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VOCABULAIRE]`, `[CHOIX]`, `[VÉRIF]`,
# `[V1]`, `[V2]`, `[SYNTHÈSE]`.

# %%
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1r" if os.path.isdir("/kaggle/working") else "sgz_a1r_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 768, 96, 64
RANK_LO, RANK_HI = 1000, 5000                    # niveau fort
MOD_MIN, MOD_MAX = 20, 500                       # niveau modéré
MID = (2, 31)
OMEGA_MIN, APPARIEMENT_MAX, ECART_NIVEAUX_MIN = 200, 0.25, 3.0
MARGE = 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1q : même témoin
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1K_MD5 = {"Qwen3-1.7B-Base": "cdc160929e703e6ec61452b0c6a742fa"}   # témoin d'A1k, 768 × 96 jetons
BRAS = ["temoin", "L_mod", "NL_mod", "L_fort", "NL_fort"]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 16, 48, 12
    RANK_LO, RANK_HI, MOD_MIN, MOD_MAX = 40, 80, 4, 30
    MID = (2, 12)
    OMEGA_MIN = 4
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-aleatoire-bis", ["smoke-qwen3-random"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
assert T0 + MID[1] <= GEN_LEN - 1
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T0} fenêtre k={MID[0]}–{MID[1]} ; niveau modéré "
      f"rang {MOD_MIN}–{MOD_MAX}, fort {RANK_LO}–{RANK_HI} ; modèles={[m[0] for m in MODELS]}")
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




def regle_v1(x, y, boots, interpretable=True):
    """F = moyenne(x) / moyenne(y) : x = anomalie à deux niveaux, y = même surprise, un seul niveau."""
    r = float(x.mean() / y.mean())
    c = ci([x[ix].mean() / y[ix].mean() for ix in boots])
    cy = ci([y[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cy[0] <= 0 <= cy[1]:
        v = "NON TESTABLE (référence non significative)"
    elif c[1] < 1:
        v = "ATTÉNUE"
    elif c[0] > 1:
        v = "AMPLIFIE"
    elif c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
        v = "ÉQUIVALENT"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "verdict": v}


def interactions(a, b, c, d):
    ma, mb, mc, md = a.mean(), b.mean(), c.mean(), d.mean()
    lr = math.log(md / mc) - math.log(mb / ma) if min(ma, mb, mc, md) > 0 else float("nan")
    return (md - mc) - (mb - ma), lr


def regle_v2(a, b, c, d, boots, prealable=None, interpretable=True):
    """a, b, c, d : L_mod, NL_mod, L_fort, NL_fort par prompt (Ω). Int = interaction additive, Lr = multiplicative."""
    it, lr = interactions(a, b, c, d)
    bs = [interactions(a[ix], b[ix], c[ix], d[ix]) for ix in boots]
    it_ci, lr_ci = ci([x for x, _ in bs]), ci([y for _, y in bs])
    surp_ci = ci([c[ix].mean() - a[ix].mean() for ix in boots])
    fm_ci, ff_ci = ci([b[ix].mean() - a[ix].mean() for ix in boots]), ci([d[ix].mean() - c[ix].mean() for ix in boots])
    out = {"Int": it, "Int_ci": it_ci, "Lr": lr, "Lr_ci": lr_ci, "surprise_ci": surp_ci, "forme_mod_ci": fm_ci,
           "forme_fort_ci": ff_ci}
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif prealable:
        v = f"NON TESTABLE ({prealable})"
    elif not surp_ci[0] > 0:
        v = "NON TESTABLE (l'effet ne croît pas avec la surprise)"
    elif b.mean() - a.mean() <= 0 or d.mean() - c.mean() <= 0 or not (fm_ci[0] > 0 or ff_ci[0] > 0):
        v = "NON TESTABLE (pas d'effet de forme positif)"
    elif lr_ci[0] > 0:
        v = "SUPRA-MULTIPLICATIF"
    elif it_ci[0] > 0:
        v = "MULTIPLICATIF" if lr_ci[0] <= 0 <= lr_ci[1] else "SUR-ADDITIF"
    elif it_ci[1] < 0:
        v = "SOUS-ADDITIF"
    elif lr_ci[1] < 0:
        v = "ADDITIF"
    else:
        v = "NON DISCRIMINANT"
    out["verdict"] = v
    return out


MULT = ("MULTIPLICATIF", "SUPRA-MULTIPLICATIF")
AU_DESSUS = MULT + ("SUR-ADDITIF",)
NON_MULT = ("ADDITIF", "SOUS-ADDITIF")


def synthese(v_par_modele, interpretable=True):
    vs = list(v_par_modele.values())
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not vs:
        return "INDÉTERMINÉE"
    if all(v in MULT for v in vs):
        return "SOUTENUE"
    if all(v in AU_DESSUS for v in vs):
        return "PARTIELLE"
    if all(v in NON_MULT for v in vs):
        return "RÉFUTÉE"
    if any(v in AU_DESSUS for v in vs) and any(v in NON_MULT for v in vs):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : relations exactes entre bras, bruit seulement sur a et c (sensibilité propre à chaque prompt)
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
sens = rng_pc.lognormal(np.log(0.03), 0.5, Pp)
a0 = sens * rng_pc.lognormal(0, 0.1, Pp)
c0 = 3 * sens * rng_pc.lognormal(0, 0.1, Pp)
f0 = 0.5 * sens


def cas(b, d):
    return regle_v2(a0, b, c0, d, bt)["verdict"]


checks = [("additif", cas(a0 + f0, c0 + f0), "ADDITIF"),
          ("multiplicatif", cas(1.5 * a0, 1.5 * c0), "MULTIPLICATIF"),
          ("supra-multiplicatif", cas(1.5 * a0, 2.25 * c0), "SUPRA-MULTIPLICATIF"),
          ("sur-additif", cas(a0 + f0, 4 / 3 * c0), "SUR-ADDITIF"),
          ("sous-additif", cas(a0 + f0, c0 + 0.3 * f0), "SOUS-ADDITIF"),
          ("sans effet de forme", cas(a0, c0), "NON TESTABLE (pas d'effet de forme positif)"),
          ("sans effet de surprise", regle_v2(a0, a0 + f0, a0, a0 + f0, bt)["verdict"],
           "NON TESTABLE (l'effet ne croît pas avec la surprise)"),
          ("préalable manquant", regle_v2(a0, a0 + f0, c0, c0 + f0, bt, "|Ω| < 200")["verdict"], "NON TESTABLE (|Ω| < 200)"),
          ("V1 amplifie", regle_v1(1.5 * c0, c0, bt)["verdict"], "AMPLIFIE"),
          ("V1 équivalent", regle_v1(c0, c0, bt)["verdict"], "ÉQUIVALENT"),
          ("V1 atténue", regle_v1(0.6 * c0, c0, bt)["verdict"], "ATTÉNUE"),
          ("synthèse soutenue", synthese({"a": "MULTIPLICATIF", "b": "SUPRA-MULTIPLICATIF"}), "SOUTENUE"),
          ("synthèse partielle", synthese({"a": "SUR-ADDITIF", "b": "MULTIPLICATIF"}), "PARTIELLE"),
          ("synthèse réfutée", synthese({"a": "ADDITIF", "b": "SOUS-ADDITIF"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "SUR-ADDITIF", "b": "ADDITIF"}), "DÉPEND DU MODÈLE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<22} : {got:<20} (attendu {att})")
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
MOT_LATIN = re.compile(r"\s?[^\W\d_]{2,}")


def ecriture(ch):
    return unicodedata.name(ch, "").split(" ")[0]


def classer_vocab(tok, n_tok, forbid, V):
    """0 : autre ; 1 : mot latin (≥ 2 lettres latines) ; 2 : non latin (lettres toutes hors latin, aucun octet incomplet)."""
    cls, ecr = np.zeros(max(n_tok, V), np.int8), {}            # au-delà de n_tok : jetons hors tokenizer, classe 0
    if isinstance(tok, SmokeTokenizer):                     # jouet : classes synthétiques, en alternance
        cls[2:n_tok] = np.where(np.arange(2, n_tok) % 2 == 0, 1, 2)
        cls[[f for f in forbid if f < len(cls)]] = 0
        return cls, {i: "JOUET" for i in range(n_tok) if cls[i] == 2}
    for i in range(n_tok):
        s = tok.decode([i])
        if not s or "�" in s:
            continue
        lettres = [ch for ch in s if unicodedata.category(ch).startswith("L")]
        if not lettres:
            continue
        latines = [ch for ch in lettres if ecriture(ch) == "LATIN"]
        if len(latines) == len(lettres) and MOT_LATIN.fullmatch(s):
            cls[i] = 1
        elif not latines and all(not ch.isdigit() for ch in s):
            cls[i] = 2
            ecr[i] = ecriture(lettres[0])
    cls[[f for f in forbid if f < len(cls)]] = 0
    return cls, ecr


def plus_proche(pos_cls, cible, exclu=()):
    """Position (rang) de pos_cls la plus proche de cible, hors exclu ; None si vide."""
    if len(pos_cls) == 0:
        return None
    o = np.argsort(np.abs(pos_cls - cible), kind="stable")
    for j in o:
        if int(pos_cls[j]) not in exclu:
            return int(pos_cls[j])
    return None


def choisir_bras(order, lp, cls, cible_fort, own):
    """order : jetons triés par logit décroissant ; lp : log-probabilités (T = 1) ; own : jeton du témoin."""
    c = cls[order]
    pos_nl, pos_l = np.nonzero(c == 2)[0], np.nonzero(c == 1)[0]
    pos_l, pos_nl = pos_l[order[pos_l] != own], pos_nl[order[pos_nl] != own]
    r_nlm = next((int(p) for p in pos_nl if p >= MOD_MIN), None)
    if r_nlm is None or r_nlm > MOD_MAX:
        return None
    r_nlf = plus_proche(pos_nl, cible_fort)
    r_lm = plus_proche(pos_l, r_nlm)
    r_lf = plus_proche(pos_l, r_nlf) if r_nlf is not None else None
    if None in (r_nlf, r_lm, r_lf) or not (RANK_LO // 2 <= r_nlf <= 2 * RANK_HI):
        return None
    rangs = {"L_mod": r_lm, "NL_mod": r_nlm, "L_fort": r_lf, "NL_fort": r_nlf}
    return {b: (int(order[r]), r, -float(lp[order[r]])) for b, r in rangs.items()}


@torch.no_grad()
def generer_temoin(g, enc, cls):
    """Témoin seul (même graine par prompt que A1c–A1q) ; au pas t₀, choix des quatre jetons dans la loi du témoin."""
    P = len(enc)
    tokm, Sg = np.zeros((P, GEN_LEN), np.int64), np.zeros((P, GEN_LEN), np.float32)
    choix = [None] * P
    cibles = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, P)
    npb = g.budget
    n_lots = math.ceil(P / npb)
    t0 = time.time()
    for b0 in range(0, P, npb):
        tl = time.time()
        rows = list(range(b0, min(P, b0 + npb)))

        def force(s, z, nxt, rows=rows):
            if s == T0:
                zz = g.masque(z.clone())
                lp = torch.log_softmax(z.double(), dim=-1).cpu().numpy()
                order = torch.argsort(zz, dim=-1, descending=True).cpu().numpy()
                for j, r in enumerate(rows):
                    choix[r] = choisir_bras(order[j], lp[j], cls, cibles[r], int(nxt[j]))
            return nxt

        S, tk = g.sample([enc[r] for r in rows], rows, T_MAIN[0], GEN_LEN, 0, force)
        tokm[rows], Sg[rows] = tk, S
        g.progress(time.time() - tl, n_lots)
    print(f"[MESURE] {g.label} génération du témoin : {time.time() - t0:.0f} s", flush=True)
    return tokm, Sg, choix


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
def run_replay(g, enc, tokm, choix, rows_om):
    P = len(rows_om)
    SR = {a: np.zeros((P, NPOS), np.float32) for a in BRAS}
    AR = {a: np.zeros((P, NPOS), np.float32) for a in BRAS}
    nb = 8
    t0 = time.time()
    for b0 in range(0, P, nb):
        sub = list(range(b0, min(P, b0 + nb)))
        rows = [rows_om[i] for i in sub]
        ct = [tokm[r, :T0 + NPOS - 1].tolist() for r in rows]
        conts = {"temoin": ct}
        for a in BRAS[1:]:
            conts[a] = [c[:T0] + [choix[r][a][0]] + c[T0 + 1:] for c, r in zip(ct, rows)]
        S, A = rejouer(g, [enc[r] for r in rows] * len(BRAS), [x for a in BRAS for x in conts[a]])
        m = len(rows)
        for ai, a in enumerate(BRAS):
            SR[a][sub], AR[a][sub] = S[ai * m:(ai + 1) * m], A[ai * m:(ai + 1) * m]
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
NIVEAUX = (("mod", "L_mod", "NL_mod"), ("fort", "L_fort", "NL_fort"))


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1r.json"), "w") as fh:
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
        tv = time.time()
        cls, ecr = classer_vocab(tok, g.n_tok, g.forbid, g.V)
        nb_ecr = {}
        for e in ecr.values():
            nb_ecr[e] = nb_ecr.get(e, 0) + 1
        print(f"[VOCABULAIRE] {label} : {g.n_tok} jetons ; mots latins {int((cls == 1).sum())} ; non latins {int((cls == 2).sum())} "
              f"({', '.join(f'{k} {v}' for k, v in sorted(nb_ecr.items(), key=lambda kv: -kv[1])[:6])}) ; {time.time() - tv:.0f} s")
        tokm, Sg, choix = generer_temoin(g, enc, cls)
        if label in A1K_MD5 and not SMOKE_TEST:
            okm = md5(tokm[:, :96]) == A1K_MD5[label]
            print(f"[ANCRAGE] {label} : témoin identique à A1k (MD5, 768 × 96) : {okm}"
                  + ("" if okm else " (écart numérique possible ; les contrastes, mesurés ici, restent valides)"))
        om = [r for r in range(len(enc)) if choix[r] is not None]
        P = len(om)
        surp = {b: np.array([choix[r][b][2] for r in om]) for b in BRAS[1:]}
        rang = {b: np.array([choix[r][b][1] for r in om]) for b in BRAS[1:]}
        app = {n: float(np.median(np.abs(surp[nl] - surp[l]))) if P else float("nan") for n, l, nl in NIVEAUX}
        ecart = float(np.median(surp["L_fort"]) - np.median(surp["L_mod"])) if P else float("nan")
        for b in BRAS[1:]:
            if P:
                print(f"[CHOIX] {label} {b:<8}: rang médian {np.median(rang[b]):.0f} ; surprise médiane {np.median(surp[b]):.2f} nats")
        print(f"[CHOIX] {label} : |Ω| = {P} prompts sur {len(enc)} ; écart médian de surprise |NL − L| : modéré "
              f"{app['mod']:.3f}, fort {app['fort']:.3f} nat (max {APPARIEMENT_MAX}) ; surprise forte − modérée {ecart:.2f} nats "
              f"(min {ECART_NIVEAUX_MIN})")
        for n, _, nl in NIVEAUX:
            e2 = {}
            for r in om:
                e = ecr.get(choix[r][nl][0], "?")
                e2[e] = e2.get(e, 0) + 1
            print(f"[CHOIX] {label} écritures de {nl} : " + ", ".join(f"{k} {v}" for k, v in sorted(e2.items(), key=lambda kv: -kv[1])[:6]))
        for r in om[:3]:
            print(f"[EXTRAIT] {label} prompt {r} : …{tok.decode(tokm[r, T0 - 8:T0].tolist())!r} → "
                  + " ; ".join(f"{b} {tok.decode([choix[r][b][0]])!r} (rang {choix[r][b][1]}, {choix[r][b][2]:.1f} nats)" for b in BRAS[1:]))
        prealable = None
        if P < OMEGA_MIN:
            prealable = f"|Ω| = {P} < {OMEGA_MIN}"
        elif max(app.values()) > APPARIEMENT_MAX:
            prealable = "appariement de surprise raté"
        elif not ecart >= ECART_NIVEAUX_MIN:
            prealable = "niveaux de surprise trop proches"
        a = {"n_omega": P, "appariement": app, "ecart_niveaux": ecart, "prealable": prealable,
             "surprise_mediane": {b: float(np.median(v)) for b, v in surp.items()} if P else {},
             "rang_median": {b: float(np.median(v)) for b, v in rang.items()} if P else {}}
        if P >= 2:
            impl = set_attention(model, "eager")
            print(f"[CONFIG] {label} : passes rejouées en attention « {impl} » (poids d'attention lus)")
            SR, AR = run_replay(g, enc, tokm, choix, om)
            dd = np.abs(SR["temoin"] - Sg[om][:, T0:T0 + NPOS])
            av = max(float(np.abs(AR[b][:, 0]).max()) for b in BRAS)
            sv = max(float(np.abs(SR[b][:, 0] - SR["temoin"][:, 0]).max()) for b in BRAS)
            print(f"[VÉRIF] {label} témoin rejoué contre généré : écart moyen {dd.mean():.1e}, max {dd.max():.1e} ; avant t₀, "
                  f"attention vers t₀ (doit être 0, masque causal) : max {av:.1e} ; entropie identique entre bras : max {sv:.1e}")
            if dd.mean() > 0.1 or av > 1e-3 or sv > 0.05:
                raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
            w = slice(MID[0], MID[1] + 1)
            M = {"A": {b: (AR[b] - AR["temoin"])[:, w].mean(1).astype(float) for b in BRAS[1:]},
                 "D": {b: (SR[b] - SR["temoin"])[:, w].mean(1).astype(float) for b in BRAS[1:]}}
            rngB = np.random.default_rng(SEED + 3)
            boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
            print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts dans Ω")
            print(f"[EFFETS] part d'attention vers t₀ dans le témoin (k = 2–31) : {AR['temoin'][:, w].mean():.4f}")
            for m, Mm in M.items():
                nom = "A, surplus d'attention vers t₀" if m == "A" else "D, effet direct sur l'entropie"
                for b in BRAS[1:]:
                    c_ = ci([Mm[b][ix].mean() for ix in boots])
                    print(f"[EFFETS] {nom} : {b:<8} {Mm[b].mean():+.5f} IC95 [{c_[0]:+.5f} ; {c_[1]:+.5f}]")
                v1 = regle_v1(Mm["NL_fort"], Mm["L_fort"], boots, ok)
                v1m = regle_v1(Mm["NL_mod"], Mm["L_mod"], boots, ok)
                v2 = regle_v2(Mm["L_mod"], Mm["NL_mod"], Mm["L_fort"], Mm["NL_fort"], boots, prealable, ok)
                print(f"[V1] {label} {m} : forme à surprise égale, niveau fort : NL/L = {v1['r']:.2f} IC95 [{v1['ci'][0]:.2f} ; "
                      f"{v1['ci'][1]:.2f}] → {v1['verdict']} ; niveau modéré (descriptif) : {v1m['r']:.2f} IC95 "
                      f"[{v1m['ci'][0]:.2f} ; {v1m['ci'][1]:.2f}] → {v1m['verdict']}")
                print(f"[V2] {label} {m} : interaction additive Int = {v2['Int']:+.5f} IC95 [{v2['Int_ci'][0]:+.5f} ; "
                      f"{v2['Int_ci'][1]:+.5f}] ; multiplicative Lr = {v2['Lr']:+.3f} IC95 [{v2['Lr_ci'][0]:+.3f} ; "
                      f"{v2['Lr_ci'][1]:+.3f}] ; effet de la surprise (L_fort − L_mod) IC95 [{v2['surprise_ci'][0]:+.5f} ; "
                      f"{v2['surprise_ci'][1]:+.5f}] → {v2['verdict']}")
                a[m] = {"moyennes": {b: float(Mm[b].mean()) for b in BRAS[1:]}, "V1": v1, "V1_mod": v1m, "V2": v2}
                prof = {b: [float((X[b] - X["temoin"])[:, lo:hi + 1].mean()) for lo, hi in ((1, 1), (2, 7), (8, 15), (16, 31))
                            if lo <= MID[1]] for b in BRAS[1:] for X in [AR if m == "A" else SR]}
                for b in BRAS[1:]:
                    print(f"[PROFIL] {label} {m} {b:<8} par tranches de k (1 | 2–7 | 8–15 | 16–31) : "
                          + " ".join(f"{x:+.4f}" for x in prof[b]))
                a[m]["profil"] = prof
            np.savez_compressed(os.path.join(OUT, f"{label}_a1r.npz"), omega=np.array(om), tok=tokm,
                                **{f"S_{b}": SR[b] for b in BRAS}, **{f"A_{b}": AR[b] for b in BRAS},
                                **{f"jeton_{b}": np.array([choix[r][b][0] for r in om]) for b in BRAS[1:]},
                                **{f"rang_{b}": rang[b] for b in BRAS[1:]}, **{f"surprise_{b}": surp[b] for b in BRAS[1:]})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1r — la complexité d'une surprise s'additionne-t-elle ou se multiplie-t-elle ?")
for m in ("A", "D"):
    v = {lab: ALL[lab][m]["V2"]["verdict"] for lab in ALL if isinstance(ALL[lab], dict) and m in ALL[lab]}
    for lab, x in v.items():
        print(f"[SYNTHÈSE] {lab:<18} {m} : V1 {ALL[lab][m]['V1']['verdict']:<12} ; V2 {x}")
    these = "NON TESTABLE (modèle manquant)" if len(v) < len(MODELS) else synthese(v, ok)
    ALL[f"these_{m}"] = these
    print(f"[SYNTHÈSE] thèse « la complexité multiplie l'intensité de la surprise », mesure {m}"
          f"{' (principale)' if m == 'A' else ' (secondaire)'} : {these}"
          + (" (prédiction SOUTENUE 10 %, PARTIELLE 10 %, RÉFUTÉE 40 %)" if m == "A" else ""))
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

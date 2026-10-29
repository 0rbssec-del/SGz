# %% [markdown]
# # SGZ A1z2 — Après un choc, le premier jeton choisi détermine-t-il l'incertitude qui suit ? (cascade, sans transplant)
#
# **Pourquoi A1z2.** A1z voulait tester si l'amplification passe par les jetons choisis après le choc. Ses bras plaçaient des
# jetons d'un contexte dans l'autre (jetons du choc derrière le jeton du témoin, et l'inverse). Or le jeton transplanté était
# lui-même un choc : surprise médiane de 16 nats dans le contexte d'arrivée, moins de 3 nats dans 1 à 3 % des prompts
# seulement. Z2 (SUFFIT) était donc un artefact, et Z1 reposait aussi sur un transplant. A1z2 ne transplante rien : tous les
# jetons forcés sont tirés dans la propre loi du modèle, dans leur propre contexte, et leur surprise est imprimée.
#
# **Dispositif.** Celui d'A1c–A1z : 768 prompts, T = 0,6, choc de rang 1000–5000 à t₀ = 64, même graine, même tirage des rangs,
# même bruit de Gumbel. Deux modèles : Qwen3-1.7B-Base et SmolLM3-3B-Base. Plan factoriel 3 × 3 par prompt, dans deux contextes :
# - **après le choc** (X à t₀) : premier jeton c₁ (le choix du modèle avec le bruit du témoin), c₁′ ou c₁″ (deux autres tirages
#   dans la même loi, bruits indépendants ; ils peuvent être égaux à c₁) ; puis suite libre avec le bruit N (celui du témoin),
#   N′ ou N″ ;
# - **après le jeton du témoin** (sans choc) : même plan, w₁, w₁′, w₁″, puis N, N′, N″.
# Un plan 2 × 2 serait trop imprécis (écart-type de ρ_a ≈ 0,033 en simulation, contre 0,014 en 3 × 3).
#
# Passes rejouées de tous les textes : entropie de Tsallis (q = 1,45) aux indices t₀ … t₀ + 31.
#
# ## Mesures
#
# y = entropie moyenne sur k = 3–31 (la loi k = 2 ne dépend que du premier jeton et du contexte : exclue de la fenêtre
# principale, gardée en descriptif). Par prompt et par contexte, avec y[i, j] (i = premier jeton, j = bruit) : analyse de
# variance à deux facteurs croisés sans répétition (carrés moyens des lignes, des colonnes et du reste, moyennés sur les
# prompts). Composantes : σ²_a = (CM_a − CM_e)/3 (premier jeton), σ²_b = (CM_b − CM_e)/3 (bruit), σ²_e = CM_e (interaction et
# reste). Part du premier jeton ρ_a = σ²_a / (σ²_a + σ²_b + σ²_e). Si chaque changement envoie la trajectoire n'importe où
# (chaos), ρ_a ≈ 0 ; si le premier jeton fixe l'incertitude qui suit, ρ_a → 1.
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# - **Préalable** : I₀ doit être AMPLIFICATION (règle F, k = 2–31, comme A1z) ; sinon Z2a NON TESTABLE.
# - **Témoin négatif** : les trois suites supplémentaires après le jeton du témoin sont d'autres tirages du même processus ;
#   leur moyenne moins le témoin doit être équivalente à 0, IC95 ⊂ [−0,3·D₀ ; +0,3·D₀] ; sinon NON INTERPRÉTABLE.
# - **Z2a (principal), le premier choix pèse-t-il plus après un choc ?** Δρ = ρ_a(choc) − ρ_a(témoin). PAS PLUS QU'APRÈS UN
#   JETON NORMAL si l'IC95 ⊂ [−0,10 ; +0,10] ; sinon PÈSE PLUS si la borne basse > 0 ; PÈSE MOINS si la borne haute < 0 ; sinon
#   INDÉTERMINÉ (même ordre que la règle F).
# - **Z2b, combien pèse-t-il après un choc ?** ρ_a(choc) : DÉTERMINANT si la borne basse ≥ 0,5 ; MODÉRÉ si l'IC95 ⊂ [0,1 ; 0,5[ ;
#   NÉGLIGEABLE si la borne haute < 0,1 ; sinon INDÉTERMINÉ.
# - **Synthèse, thèse « l'amplification passe par le premier jeton choisi après le choc »** : SOUTENUE si Z2a = PÈSE PLUS et
#   Z2b ∈ {DÉTERMINANT, MODÉRÉ} pour les deux modèles ; RÉFUTÉE si Z2a ∈ {PAS PLUS, PÈSE MOINS} pour les deux ; DÉPEND DU
#   MODÈLE si un modèle suit le premier schéma et l'autre le second ; sinon INDÉTERMINÉE.
# - **Descriptifs** : surprise de c₁, c₁′, w₁, w₁′ dans leur propre loi (aucun transplant) ; part des tirages où c₁′ = c₁ et
#   w₁′ = w₁ (trois tirages identiques) ; ρ_a restreint aux prompts où au moins deux premiers jetons diffèrent ; ρ_b ;
#   composantes en valeur absolue ; fenêtre k = 2–31 ; ρ_a par tranches de k ; amplification estimée sur les neuf suites après
#   le choc.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Z2a : PÈSE PLUS 45 %, PAS PLUS 35 %, INDÉTERMINÉ 20 %. Z2b : MODÉRÉ 50 %, NÉGLIGEABLE 25 %, DÉTERMINANT 5 %. Synthèse
# SOUTENUE 30 %, RÉFUTÉE 30 %. ρ_a(témoin) < 0,2 : 70 %. Premiers tirages différents plus fréquents après le choc qu'après le
# témoin : 85 %.
#
# ## Limites connues
#
# - Trois niveaux par facteur : les composantes de variance sont estimées prompt par prompt avec peu de degrés de liberté (la
#   précision vient des 768 prompts).
# - La part ρ_a dépend de la variance totale, qui peut différer entre contextes : σ²_a est aussi imprimé en valeur absolue.
# - Une bande de chocs (rang 1000–5000), une température, deux modèles ; anomalies artificielles.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~30 min pour Qwen3,
# ~40 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[SURPRISE]`, `[TÉMOIN NÉGATIF]`,
# `[F]`, `[Z2a]`, `[Z2b]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1z2" if os.path.isdir("/kaggle/working") else "sgz_a1z2_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T1 = 768, 96, 64
RANK_LO, RANK_HI = 1000, 5000
MID = (2, 31)                                    # fenêtre de la règle F (comme A1c–A1z)
FEN = (3, 31)                                    # fenêtre principale du plan factoriel
MARGE_REL = 0.3
MARGE_DRHO = 0.10
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1z
FRAIS = SEED + 7919                              # bruit N′ des suites
ALT, ALT2 = SEED + 4241, SEED + 2 * 4241         # bruits des autres tirages du premier jeton (c₁′, c₁″ ; w₁′, w₁″)
FRAIS2 = SEED + 2 * 7919                         # bruit N″ des suites
PREMIERS, BRUITS = [SEED, ALT, ALT2], [SEED, FRAIS, FRAIS2]
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1Z = {"Qwen3-1.7B-Base": {"D": 0.07734, "I": 0.03958}, "SmolLM3-3B-Base": {"D": 0.07512, "I": 0.04282}}
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1 = int(os.environ.get("SMOKE_N", "16")), 48, 12
    RANK_LO, RANK_HI = 20, 80
    MID, FEN = (2, 12), (3, 12)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
NSUITE = NPOS - 2                                # jetons après t₀ : t₀ + 1 … t₀ + 30
LONG = T1 + 1 + NSUITE                           # textes rejoués : jetons 0 … t₀ + 30
assert LONG <= GEN_LEN
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T1} fenêtre F k={MID[0]}–{MID[1]}, plan factoriel "
      f"k={FEN[0]}–{FEN[1]} ; modèles={[m[0] for m in MODELS]}")
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


def carres(Y):
    """Y : (P, I, J), y[p, i, j] (i = premier jeton, j = bruit). Carrés moyens par prompt : premier jeton, bruit, reste."""
    I, J = Y.shape[1:]
    m = Y.mean((1, 2), keepdims=True)
    ri, cj = Y.mean(2, keepdims=True) - m, Y.mean(1, keepdims=True) - m
    res = Y - m - ri - cj
    return np.stack([J * (ri ** 2).sum((1, 2)) / (I - 1), I * (cj ** 2).sum((1, 2)) / (J - 1),
                     (res ** 2).sum((1, 2)) / ((I - 1) * (J - 1))], 1)


def parts(K, I=3, J=3):
    """K : (P, 3) carrés moyens. Composantes σ²_a (premier jeton), σ²_b (bruit), σ²_e (reste) et parts ρ_a, ρ_b."""
    msa, msb, mse = K.mean(0)
    sa, sb = (msa - mse) / J, (msb - mse) / I
    tot = sa + sb + mse
    return {"rho_a": float(sa / tot), "rho_b": float(sb / tot), "s2_a": float(sa), "s2_b": float(sb), "s2_e": float(mse)}


def regle_z2a(Kc, Kt, boots, amp, interpretable=True):
    rc, rt = parts(Kc)["rho_a"], parts(Kt)["rho_a"]
    dci = ci([parts(Kc[ix])["rho_a"] - parts(Kt[ix])["rho_a"] for ix in boots])
    m = MARGE_DRHO
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif not amp:
        v = "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"
    elif -m <= dci[0] and dci[1] <= m:
        v = "PAS PLUS QU'APRÈS UN JETON NORMAL"
    elif dci[0] > 0:
        v = "PÈSE PLUS APRÈS UN CHOC"
    elif dci[1] < 0:
        v = "PÈSE MOINS APRÈS UN CHOC"
    else:
        v = "INDÉTERMINÉ"
    return {"rho_choc": rc, "rho_temoin": rt, "d": rc - rt, "ci": dci, "verdict": v}


def regle_z2b(Kc, boots, interpretable=True):
    r = parts(Kc)["rho_a"]
    c = ci([parts(Kc[ix])["rho_a"] for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] >= 0.5:
        v = "DÉTERMINANT"
    elif c[0] >= 0.1 and c[1] < 0.5:
        v = "MODÉRÉ"
    elif c[1] < 0.1:
        v = "NÉGLIGEABLE"
    else:
        v = "INDÉTERMINÉ"
    return {"rho": r, "ci": c, "verdict": v}


def synthese(za, zb, interpretable=True):
    """za, zb : dict modèle → verdict."""
    ms = list(za)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    oui = [m for m in ms if za[m] == "PÈSE PLUS APRÈS UN CHOC" and zb.get(m) in ("DÉTERMINANT", "MODÉRÉ")]
    non = [m for m in ms if za[m] in ("PAS PLUS QU'APRÈS UN JETON NORMAL", "PÈSE MOINS APRÈS UN CHOC")]
    if len(oui) == len(ms):
        return "SOUTENUE"
    if len(non) == len(ms):
        return "RÉFUTÉE"
    if oui and non and len(oui) + len(non) == len(ms):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : plans 2 × 2 synthétiques, composantes de variance connues
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]


def plan(sa, sb, se):
    f, gn, e = rng_pc.normal(0, 1, (Pp, 3, 1)), rng_pc.normal(0, 1, (Pp, 1, 3)), rng_pc.normal(0, 1, (Pp, 3, 3))
    return carres(sa * f + sb * gn + se * e)


fort, faible, nul_ = plan(1.0, 0.3, 0.5), plan(0.3, 0.3, 1.0), plan(0.05, 0.3, 1.0)
checks = [("Z2a pèse plus", regle_z2a(fort, faible, bt, True)["verdict"], "PÈSE PLUS APRÈS UN CHOC"),
          ("Z2a pas plus", regle_z2a(nul_, plan(0.05, 0.3, 1.0), bt, True)["verdict"], "PAS PLUS QU'APRÈS UN JETON NORMAL"),
          ("Z2a pèse moins", regle_z2a(faible, fort, bt, True)["verdict"], "PÈSE MOINS APRÈS UN CHOC"),
          ("Z2a petit écart", regle_z2a(plan(0.4, 0.3, 1.0), faible, bt, True)["verdict"], "PAS PLUS QU'APRÈS UN JETON NORMAL"),
          ("Z2a sans amplification", regle_z2a(fort, faible, bt, False)["verdict"], "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"),
          ("Z2b déterminant", regle_z2b(fort, bt)["verdict"], "DÉTERMINANT"),
          ("Z2b modéré", regle_z2b(plan(0.6, 0.3, 0.8), bt)["verdict"], "MODÉRÉ"),
          ("Z2b négligeable", regle_z2b(nul_, bt)["verdict"], "NÉGLIGEABLE"),
          ("nul équivalent", regle_nul((lambda x: x - x.mean())(rng_pc.normal(0, 0.03, Pp)), 0.02, bt)["verdict"], "ÉQUIVALENT À 0"),
          ("nul biaisé", regle_nul(0.05 + rng_pc.normal(0, 0.15, Pp), 0.02, bt)["verdict"], "BIAISÉ"),
          ("synthèse soutenue", synthese({"a": "PÈSE PLUS APRÈS UN CHOC", "b": "PÈSE PLUS APRÈS UN CHOC"},
                                         {"a": "MODÉRÉ", "b": "DÉTERMINANT"}), "SOUTENUE"),
          ("synthèse réfutée", synthese({"a": "PAS PLUS QU'APRÈS UN JETON NORMAL", "b": "PÈSE MOINS APRÈS UN CHOC"},
                                        {"a": "NÉGLIGEABLE", "b": "MODÉRÉ"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "PÈSE PLUS APRÈS UN CHOC", "b": "PAS PLUS QU'APRÈS UN JETON NORMAL"},
                                       {"a": "MODÉRÉ", "b": "NÉGLIGEABLE"}), "DÉPEND DU MODÈLE"),
          ("synthèse indéterminée", synthese({"a": "PÈSE PLUS APRÈS UN CHOC", "b": "PÈSE PLUS APRÈS UN CHOC"},
                                             {"a": "NÉGLIGEABLE", "b": "MODÉRÉ"}), "INDÉTERMINÉE")]
print(f"[TÉMOIN POSITIF] parts synthétiques : fort ρ_a = {parts(fort)['rho_a']:.2f} (attendu 0,75), faible {parts(faible)['rho_a']:.2f} "
      f"(0,08), nul {parts(nul_)['rho_a']:.3f} (0,002)")
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<42} (attendu {att})")
POSITIVE_CONTROL_OK = all(g == e for _, g, e in checks)
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON INTERPRÉTABLE'}")

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
def tirer_premier(g, enc, idx, pref, base, autres=()):
    """Jeton de la position p = pref.shape[1], tiré comme dans Gen.sample au pas p avec le bruit de graine « base », et surprises
    −log p (loi brute du modèle, dans ce contexte) de ce jeton et des jetons « autres » (tableaux (P,)) à cette position."""
    P, p = pref.shape
    tir, s_t = np.zeros(P, np.int64), np.zeros(P)
    s_o = [np.zeros(P) for _ in autres]
    head = g.model.get_output_embeddings()
    for b0 in range(0, P, g.budget):
        loc = list(range(b0, min(P, b0 + g.budget)))
        ids, attn, _ = g.left_pad([enc[int(idx[i])] + pref[i].tolist() for i in loc])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        h = g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        z = head(h.to(head.weight.device)).float()
        seeds = [base + 100003 * int(idx[i]) + p for i in loc]
        nxt = torch.argmax(g.masque(z / T_MAIN[0] + gumbel_noise(g.V, seeds, z.device)), dim=-1)
        lp = torch.log_softmax(z.double(), dim=-1)
        for j, i in enumerate(loc):
            tir[i] = int(nxt[j])
            s_t[i] = -float(lp[j, tir[i]])
            for o, so in zip(autres, s_o):
                so[i] = -float(lp[j, int(o[i])])
    return tir, s_t, s_o

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
W = slice(MID[0], MID[1] + 1)
WF = slice(FEN[0], FEN[1] + 1)
TRANCHES = [(lo, min(hi, MID[1])) for lo, hi in ((3, 7), (8, 15), (16, 31)) if lo <= MID[1]]


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1z2.json"), "w") as fh:
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
        pX, pW = np.concatenate([pre, X[:, None]], 1), tokm[:, :T1 + 1]
        tg = time.time()
        rep = generer_depuis(g, enc, idx, pW)
        same_1 = float(np.mean(rep[:, T1 + 1] == tokm[:, T1 + 1]))
        same = float(np.mean((rep[:, T1 + 1:LONG] == tokm[:, T1 + 1:LONG]).all(1)))
        print(f"[VÉRIF] {label} reprise (jeton du témoin, même bruit) : premier jeton identique au témoin {same_1:.1%} ; suite "
              f"entière identique {same:.1%} (en fp16, un quasi-ex aequo peut basculer sans erreur)")
        print(f"[PROGRESSION] {label} suites libres : estimation ~{(time.time() - tg) * 17 / 60:.0f} min pour 17 suites", flush=True)
        if same_1 < 0.9 or same < 0.2:
            raise RuntimeError("la génération libre ne redonne pas le témoin : chemin de génération défectueux")
        textes = {"temoin": tokm[:, :LONG].copy()}
        textes["rejoue"] = textes["temoin"].copy()
        textes["rejoue"][:, T1] = X
        textes["c0n0"] = generer_depuis(g, enc, idx, pX)
        c1, w1 = textes["c0n0"][:, T1 + 1], tokm[:, T1 + 1]

        # Premiers jetons : tirages dans la propre loi du modèle, dans leur propre contexte (aucun transplant)
        PC, PW, SC, SW = [], [], [], []
        for b in PREMIERS:
            tc, sc, (s_c1,) = tirer_premier(g, enc, idx, pX, b, (c1,))
            tw, sw, (s_w1,) = tirer_premier(g, enc, idx, pW, b, (w1,))
            if b == SEED:
                ok_c, ok_w = float(np.mean(tc == c1)), float(np.mean(tw == w1))
                print(f"[VÉRIF] {label} retirage du premier jeton avec le bruit du témoin : identique au choix du choc {ok_c:.1%}, "
                      f"au choix du témoin {ok_w:.1%}")
                if min(ok_c, ok_w) < 0.9:
                    raise RuntimeError("le tirage du premier jeton ne reproduit pas la génération : graines désalignées")
                tc, tw, sc, sw = c1, w1, s_c1, s_w1
            PC.append(tc)
            PW.append(tw)
            SC.append(sc)
            SW.append(sw)
        for i in range(3):
            for j, b in enumerate(BRUITS):
                if i or j:
                    textes[f"c{i}n{j}"] = generer_depuis(g, enc, idx, np.concatenate([pX, PC[i][:, None]], 1), base=b)
                    textes[f"w{i}n{j}"] = generer_depuis(g, enc, idx, np.concatenate([pW, PW[i][:, None]], 1), base=b)
        print(f"[MESURE] {label} suites libres : {time.time() - tg:.0f} s", flush=True)
        for i in range(3):
            for j in range(3):
                if i or j:
                    assert (textes[f"c{i}n{j}"][:, :T1 + 1] == pX).all() and (textes[f"c{i}n{j}"][:, T1 + 1] == PC[i]).all()
                    assert (textes[f"w{i}n{j}"][:, :T1 + 1] == pW).all() and (textes[f"w{i}n{j}"][:, T1 + 1] == PW[i]).all()
        tr = time.time()
        SR = {k: rejouer_S(g, enc, idx, v) for k, v in textes.items()}
        SR["w0n0"] = SR["temoin"]
        print(f"[MESURE] {label} passes rejouées ({len(textes)} textes) : {time.time() - tr:.0f} s", flush=True)
        dd = np.abs(SR["temoin"] - res["temoin"]["S"][:, T1:T1 + NPOS])
        sv = max(float(np.abs(SR[k][:, 0] - SR["temoin"][:, 0]).max()) for k in SR)
        s1c = max(float(np.abs(SR[f"c{i}n{j}"][:, 1] - SR["rejoue"][:, 1]).max()) for i in range(3) for j in range(3))
        print(f"[VÉRIF] {label} : témoin rejoué contre généré, écart moyen {dd.mean():.1e} (max {dd.max():.1e}) ; entropie de la "
              f"loi du jeton t₀ identique entre textes : max {sv:.1e} ; loi du jeton t₀ + 1 identique entre les suites du choc : "
              f"max {s1c:.1e}")
        if dd.mean() > 0.1 or sv > 0.05 or s1c > 0.05:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")

        moy = lambda k, w: SR[k][:, w].mean(1).astype(float)
        St, Sr, Sc = moy("temoin", W), moy("rejoue", W), moy("c0n0", W)
        D0, I0 = Sr - St, Sc - Sr
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        cim = lambda x: ci([x[ix].mean() for ix in boots])
        F0 = regle_F(D0, I0, boots, ok)
        amp = F0["verdict"] == "AMPLIFICATION"
        Tn = np.mean([moy(f"w{i}n{j}", W) for i in range(3) for j in range(3) if i or j], 0) - St
        nul = regle_nul(Tn, MARGE_REL * abs(float(D0.mean())), boots, ok)
        t_ok = ok and nul["verdict"] == "ÉQUIVALENT À 0"
        Y = lambda c, w: np.stack([np.stack([moy(f"{c}{i}n{j}", w) for j in range(3)], 1) for i in range(3)], 1)
        Kc, Kt = carres(Y("c", WF)), carres(Y("w", WF))
        z2a = regle_z2a(Kc, Kt, boots, amp, t_ok)
        z2b = regle_z2b(Kc, boots, t_ok)
        pc, pt = parts(Kc), parts(Kt)
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts")
        print(f"[F] {label} : D₀ = {fmt(F0['D'], F0['D_ci'])} ; I₀ = {fmt(F0['I'], F0['I_ci'])} → {F0['verdict']}"
              + (f" (A1z : D {A1Z[label]['D']:+.4f}, I {A1Z[label]['I']:+.4f})" if label in A1Z else ""))
        print(f"[TÉMOIN NÉGATIF] {label} 8 autres suites après le jeton du témoin − témoin = {fmt(nul['m'], nul['ci'])} ; marge "
              f"±{nul['marge']:.4f} → {nul['verdict']}" + ("" if t_ok or not ok else " : Z2a et Z2b NON INTERPRÉTABLES"))
        med = lambda v: float(np.median(v))
        ident_c = float(np.mean((PC[0] == PC[1]) & (PC[0] == PC[2])))
        ident_w = float(np.mean((PW[0] == PW[1]) & (PW[0] == PW[2])))
        print(f"[SURPRISE] {label} premiers jetons dans leur propre loi (aucun transplant) : après le choc c₁ {med(SC[0]):.2f}, c₁′ "
              f"{med(SC[1]):.2f}, c₁″ {med(SC[2]):.2f} nats ; après le témoin w₁ {med(SW[0]):.2f}, w₁′ {med(SW[1]):.2f}, w₁″ "
              f"{med(SW[2]):.2f} nats (médianes) ; trois tirages identiques : après le choc {ident_c:.1%}, après le témoin {ident_w:.1%}")
        for nom, p_ in (("choc", pc), ("témoin", pt)):
            print(f"[COMPOSANTES] {label} après le {nom:<6} (k = {FEN[0]}–{FEN[1]}) : σ²_a = {p_['s2_a']:.2e} (premier jeton), σ²_b = "
                  f"{p_['s2_b']:.2e} (bruit), σ²_e = {p_['s2_e']:.2e} (reste) ; ρ_a = {p_['rho_a']:.3f}, ρ_b = {p_['rho_b']:.3f}")
        print(f"[Z2a] {label} Δρ = ρ_a(choc) − ρ_a(témoin) = {z2a['rho_choc']:.3f} − {z2a['rho_temoin']:.3f} = {fmt(z2a['d'], z2a['ci'], '+.3f')} "
              f"→ {z2a['verdict']}")
        print(f"[Z2b] {label} ρ_a(choc) = {fmt(z2b['rho'], z2b['ci'], '.3f')} → {z2b['verdict']}")
        a = {"F0": F0, "temoin_negatif": nul, "reprise": {"premier": same_1, "suite": same}, "Z2a": z2a, "Z2b": z2b,
             "parts_choc": pc, "parts_temoin": pt, "identiques": {"choc": ident_c, "temoin": ident_w},
             "surprise_mediane": {"c": [med(s) for s in SC], "w": [med(s) for s in SW]}}
        dif_c = ~((PC[0] == PC[1]) & (PC[0] == PC[2]))
        dif_w = ~((PW[0] == PW[1]) & (PW[0] == PW[2]))
        rc_d, rt_d = parts(Kc[dif_c]), parts(Kt[dif_w])
        k2c, k2t = parts(carres(Y("c", W))), parts(carres(Y("w", W)))
        a["descriptifs"] = {"rho_a_premiers_differents": {"choc": rc_d["rho_a"], "temoin": rt_d["rho_a"]},
                            "rho_a_k2_31": {"choc": k2c["rho_a"], "temoin": k2t["rho_a"]}}
        print(f"[DESCRIPTIF] {label} ρ_a quand au moins deux premiers jetons diffèrent : choc {rc_d['rho_a']:.3f} ({dif_c.mean():.0%} des "
              f"prompts), témoin {rt_d['rho_a']:.3f} ({dif_w.mean():.0%}) ; fenêtre k = 2–31 : choc {k2c['rho_a']:.3f}, témoin "
              f"{k2t['rho_a']:.3f}")
        prof = {}
        for nom, c in (("choc", "c"), ("temoin", "w")):
            prof[nom] = [parts(carres(Y(c, slice(lo, hi + 1))))["rho_a"] for lo, hi in TRANCHES]
        a["profil_rho_a"] = prof
        print(f"[PROFIL] {label} ρ_a par tranches de k ({' | '.join(f'{lo}–{hi}' for lo, hi in TRANCHES)}) : choc "
              + "  ".join(f"{x:.3f}" for x in prof["choc"]) + " ; témoin " + "  ".join(f"{x:.3f}" for x in prof["temoin"]))
        I9 = np.mean([moy(f"c{i}n{j}", W) for i in range(3) for j in range(3)], 0) - Sr
        a["I_neuf_suites"] = {"m": float(I9.mean()), "ci": cim(I9)}
        print(f"[DESCRIPTIF] {label} amplification estimée sur les neuf suites après le choc : I = {fmt(I9.mean(), cim(I9))} "
              f"(une seule suite : {I0.mean():+.4f})")
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1z2.npz"), PC=np.stack(PC), PW=np.stack(PW), SC=np.stack(SC),
                            SW=np.stack(SW), **{f"S_{k}": v for k, v in SR.items() if k != "w0n0"},
                            **{f"tok_{k}": v for k, v in textes.items()})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1z2 — après un choc, le premier jeton choisi détermine-t-il l'incertitude qui suit ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "Z2a" in ALL[m]]
v = {k: {m: ALL[m][k]["verdict"] for m in labs} for k in ("Z2a", "Z2b")}
for m in labs:
    print(f"[SYNTHÈSE] {m:<18} : Z2a {v['Z2a'][m]:<34} ; Z2b {v['Z2b'][m]}")
these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese(v["Z2a"], v["Z2b"], ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « l'amplification passe par le premier jeton choisi après le choc » : {these} "
      f"(prédiction SOUTENUE 30 %, RÉFUTÉE 30 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

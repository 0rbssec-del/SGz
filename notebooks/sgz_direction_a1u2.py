# %% [markdown]
# # SGZ A1u2 — Reprise de V4 : la direction v₁ porte-t-elle l'amplification, à dégât égal ?
#
# **Point de départ (A1u, 29 oct.).** Sur Qwen3-1.7B-Base, retirer v₁ (premier axe des écarts choc − témoin à la couche
# 15) du flux résiduel réduit l'effet de la trajectoire I de 82 % [17 ; 122], sous les 20 témoins aléatoires : verdict
# CONTRIBUTION SPÉCIFIQUE. Mais le modèle est cassé : l'entropie du témoin tombe de 0,74 à 0,07, hors de la plage des
# témoins (0,22 à 0,79). Le dégât n'était pas apparié (même défaut que A1q). Une baisse de I dans un modèle cassé ne dit
# rien du mécanisme.
#
# **Ce qui change.** Deux sources de dégât sont retirées :
# - **ablation par la moyenne** au lieu de zéro : ⟨h, v⟩ est remplacé par sa moyenne m sur le témoin de calibration, pas
#   par 0 (un axe qui porte une grande moyenne sort le modèle de sa distribution quand on le met à zéro) ;
# - **ablation à partir de t₀ seulement** (le jeton choqué et la suite), plus à toutes les positions : le prompt et le
#   début du texte restent intacts, dans le témoin comme dans le choc.
#
# Le dégât est ensuite **contrôlé** : le verdict exige que celui de v₁ tombe dans la plage de celui des témoins aléatoires.
#
# **Dispositif.** Celui d'A1u : 768 prompts, T = 0,6, 96 jetons, choc de rang 1000–5000 à t₀ = 64, même graine. Couche
# l* = 15 et v₁ = celui d'A1u (embarqué ici ; recalculé sur la calibration pour vérification). Calibration sur les prompts
# pairs (moyenne m, axes principaux), test sur les impairs (384). Pour chaque condition, témoin et choc sont régénérés à
# partir de t₀ (le début, intact, est repris de la génération sans ablation ; même bruit, même jeton choqué), puis passes
# rejouées : témoin, choc, rejoué. Conditions :
# - aucune ;
# - **v₁, moyenne, à partir de t₀** (principale) ;
# - v₁, moyenne, à partir de t₀, retiré à moitié (repli si le dégât de v₁ entier n'est pas apparié) ;
# - v₁, zéro, à partir de t₀ (descriptif : sépare l'effet du mode de celui des positions) ;
# - 24 directions aléatoires, moyenne, à partir de t₀ : tirées dans les 64 premiers axes principaux du témoin à l*
#   (pondérées par leur variance), orthogonales à v₁ ; les 20 premières sont celles d'A1u.
#
# ## Critères (fixés avant toute exécution ; bootstrap sur les prompts de test, B = 1000)
#
# - Effet de la trajectoire I et effet direct D par condition (règle F de A1c, k = 2–31).
# - **Dégât** d(c) = entropie moyenne du témoin (k = 2–31) sous la condition c − sans ablation.
# - **V4b (principal)** : NON TESTABLE si I(aucune) n'est pas AMPLIFICATION ; NON TESTABLE (DÉGÂT HORS DES TÉMOINS) si d(v₁)
#   sort de [min ; max] des d des 24 témoins ; sinon règle d'A1u sur r = 1 − I(v₁) / I(aucune) : MÉCANISME PRINCIPAL si
#   borne basse de IC95(r) ≥ 0,5 et I(v₁) inférieur aux 24 témoins ; CONTRIBUTION SPÉCIFIQUE si borne basse > 0 et I(v₁)
#   inférieur aux 24 témoins ; NON SPÉCIFIQUE si borne basse > 0 sans cela ; AUGMENTE si borne haute < 0 ; sinon PAS D'EFFET.
#   (Sous l'hypothèse nulle, « inférieur aux 24 témoins » arrive avec une probabilité de 1/25 = 4 %.)
# - **Repli** : si V4b est NON TESTABLE à cause du dégât, la même règle s'applique à v₁ retiré à moitié
#   (⟨h, v⟩ → m + (⟨h, v⟩ − m) / 2), si son dégât est, lui, dans la plage ; le verdict porte alors la mention « repli ».
# - **Synthèse, thèse « à dégât égal, v₁ porte une part spécifique de l'amplification »** : SOUTENUE si MÉCANISME PRINCIPAL
#   ou CONTRIBUTION SPÉCIFIQUE ; RÉFUTÉE si NON SPÉCIFIQUE, PAS D'EFFET ou AUGMENTE ; NON TESTABLE si le dégât n'est apparié
#   ni pour v₁ entier ni pour le repli ; sinon INDÉTERMINÉE. Calculée sur V4b, ou sur le repli si V4b n'est pas apparié.
# - **Descriptifs** : rang de I(v₁) parmi les témoins ; I ajusté au dégât (droite I ~ d sur les 24 témoins, résidu de v₁
#   rapporté à la dispersion des résidus ; extrapolation si v₁ sort de la plage) ; v₁ zéro (dégât, D, I) ; |m| / écart-type
#   de la projection pour v₁ et pour les témoins (v₁ porte-t-il une grande moyenne ?) ; part des suites du témoin changées
#   par l'ablation ; reproduction d'A1u (l*, cos entre v₁ recalculé et v₁ d'A1u, I sans ablation).
# - Témoins positifs synthétiques pour chaque règle ; vérification du crochet (projection à m après t₀, aucun écart avant).
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Dégât de v₁ en moyenne dans la plage des témoins : 60 %. V4b : MÉCANISME PRINCIPAL 10 %, CONTRIBUTION SPÉCIFIQUE 25 %,
# NON SPÉCIFIQUE 25 %, PAS D'EFFET 25 %, NON TESTABLE (dégât) 15 % (repli utilisé : 40 %). v₁ zéro à partir de t₀ : entropie du témoin sous 0,3
# (dégât fort) : 65 %. |m| / écart-type plus grand pour v₁ que pour tous les témoins : 60 %.
#
# ## Limites connues
#
# - Une couche, une direction ; un mécanisme réparti apparaîtrait comme NON SPÉCIFIQUE ou PAS D'EFFET.
# - L'ablation par la moyenne retire la variation selon v₁, pas sa moyenne : c'est bien la composante du choc selon v₁
#   (δ = h(rejoué) − h(témoin)) qui disparaît, dans le choc comme dans le témoin.
# - Appariement sur un seul critère de dégât (l'entropie du témoin) ; d'autres dégâts (sens du texte) ne sont pas mesurés.
# - Un seul modèle (Qwen3-1.7B-Base, celui d'A1u).
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~40 min [estimation].
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[ANCRAGE]`, `[VÉRIF]`, `[DÉGÂT]`, `[V4b]`, `[SYNTHÈSE]`.

# %%
import base64
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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1u2" if os.path.isdir("/kaggle/working") else "sgz_a1u2_out")
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
N_ALEA, N_PC = 24, 64
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1u
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1U = {"Qwen3-1.7B-Base": {"l_star": 15, "I_test": 0.04276, "v1_md5": "efc3a44846f6914b9ec87f5d2b174188"}}
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1 = 16, 48, 12
    RANK_LO, RANK_HI = 20, 80
    MID = (2, 12)
    N_ALEA, N_PC = 4, 8
    BOOT_N = 50
    N_PROMPTS = int(os.environ.get("SMOKE_N", "16"))
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
assert T1 + MID[1] <= GEN_LEN - 1
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T1} fenêtre k={MID[0]}–{MID[1]} ; {N_ALEA} directions "
      f"aléatoires dans {N_PC} axes ; ablation par la moyenne à partir de t₀ ; modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# v₁ d'A1u (Qwen3-1.7B-Base, couche 15), float32 petit-boutiste, 2048 valeurs, encodé en base64
V1_A1U_B64 = {"Qwen3-1.7B-Base": (
    "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"
    "0XYBY8xPQpvNMuALvD67A8WWF8PJSuljw=")}

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


def regle_v4b(I0, Iv, I_alea, d_v, d_alea, boots, verdict_F0, interpretable=True):
    """I0, Iv : I par prompt sans ablation et avec v₁ ; I_alea, d_alea : I moyen et dégât des témoins ; d_v : dégât de v₁."""
    r = float(1 - Iv.mean() / I0.mean())
    c = ci([1 - Iv[ix].mean() / I0[ix].mean() for ix in boots])
    sous_tous = bool(Iv.mean() < min(I_alea)) if len(I_alea) else False
    apparie = bool(len(d_alea) and min(d_alea) <= d_v <= max(d_alea))
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif verdict_F0 != "AMPLIFICATION":
        v = f"NON TESTABLE (sans ablation : {verdict_F0})"
    elif not apparie:
        v = "NON TESTABLE (DÉGÂT HORS DES TÉMOINS)"
    elif c[0] >= 0.5 and sous_tous:
        v = "MÉCANISME PRINCIPAL"
    elif c[0] > 0 and sous_tous:
        v = "CONTRIBUTION SPÉCIFIQUE"
    elif c[0] > 0:
        v = "NON SPÉCIFIQUE"
    elif c[1] < 0:
        v = "AUGMENTE"
    else:
        v = "PAS D'EFFET"
    return {"r": r, "ci": c, "sous_tous_les_temoins": sous_tous, "apparie": apparie, "verdict": v}


def synthese(v4b, interpretable=True):
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if v4b in ("MÉCANISME PRINCIPAL", "CONTRIBUTION SPÉCIFIQUE"):
        return f"SOUTENUE ({v4b.lower()})"
    if v4b in ("NON SPÉCIFIQUE", "PAS D'EFFET", "AUGMENTE"):
        return "RÉFUTÉE"
    if v4b.startswith("NON TESTABLE"):
        return v4b
    return "INDÉTERMINÉE"


def ajuste(I_v, d_v, I_alea, d_alea):
    """I de v₁ rapporté à la droite I ~ d des témoins : résidu, résidu / écart-type des résidus, extrapolation ou non."""
    d_alea, I_alea = np.asarray(d_alea, float), np.asarray(I_alea, float)
    b, a = np.polyfit(d_alea, I_alea, 1)
    res = I_alea - (a + b * d_alea)
    rv = I_v - (a + b * d_v)
    return {"pente": float(b), "residu": float(rv), "z": float(rv / max(res.std(ddof=1), 1e-12)),
            "rang": int((res < rv).sum()), "extrapolation": bool(not (d_alea.min() <= d_v <= d_alea.max()))}


# Témoins positifs synthétiques
rng_pc = np.random.default_rng(SEED + 1)
Pp = 384
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
I0_pc = 0.05 + rng_pc.normal(0, 0.1, Pp)
bruit = lambda: (lambda x: x - x.mean())(rng_pc.normal(0, 0.02, Pp))   # centré : ne déplace pas la moyenne
Ia, da = [0.05, 0.048, 0.051, 0.052], [-0.02, 0.01, -0.05, 0.03]
v4 = lambda Iv, I_al=Ia, d_v=0.0, F0="AMPLIFICATION": regle_v4b(I0_pc, Iv, I_al, d_v, da, bt, F0)["verdict"]
aj = ajuste(0.01, 0.0, [0.05 + 0.2 * d + e for d, e in zip(da, [0.001, -0.001, 0.0005, -0.0005])], da)
checks = [("mécanisme principal", v4(0.2 * I0_pc + bruit()), "MÉCANISME PRINCIPAL"),
          ("contribution", v4(0.7 * I0_pc + bruit()), "CONTRIBUTION SPÉCIFIQUE"),
          ("non spécifique", v4(0.7 * I0_pc + bruit(), [0.03, 0.048, 0.051, 0.052]), "NON SPÉCIFIQUE"),
          ("pas d'effet", v4(I0_pc + bruit()), "PAS D'EFFET"),
          ("augmente", v4(1.6 * I0_pc + bruit()), "AUGMENTE"),
          ("dégât hors des témoins", v4(0.2 * I0_pc + bruit(), d_v=-0.6), "NON TESTABLE (DÉGÂT HORS DES TÉMOINS)"),
          ("sans amplification", v4(0.2 * I0_pc, F0="ÉQUIVALENT"), "NON TESTABLE (sans ablation : ÉQUIVALENT)"),
          ("synthèse soutenue", synthese("CONTRIBUTION SPÉCIFIQUE"), "SOUTENUE (contribution spécifique)"),
          ("synthèse réfutée", synthese("PAS D'EFFET"), "RÉFUTÉE"),
          ("ajustement : résidu", "très négatif" if aj["z"] < -10 and aj["rang"] == 0 else f"{aj['z']:+.1f}", "très négatif")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<38} (attendu {att})")
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
# ---------- Génération (identique à A1c–A1u ; crochet d'ablation actif à partir de t₀) ----------
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
        self.proj = None                         # crochet d'ablation, actif à partir de t₀

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
            if self.proj is not None:                # l'entrée du pas s est le jeton step0 + s − 1
                self.proj.actif, self.proj.p0 = step0 + s >= T1 + 1, 0
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
        if self.proj is not None:
            self.proj.actif = False
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
# ---------- Choc à t₀ (comme A1u), crochets, passes rejouées, suites régénérées à partir de t₀ ----------
def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque


def sortie(out):
    return out[0] if isinstance(out, tuple) else out


class Projecteur:
    """Ablation de la composante selon v (unitaire) à la sortie de la couche l, aux positions p0 et suivantes de l'appel en
    cours, quand actif : ⟨h, v⟩ est remplacé par m + (1 − α)(⟨h, v⟩ − m) (m = moyenne du témoin : ablation par la moyenne ;
    m = 0 : par zéro ; α = 1 : retrait complet)."""

    def __init__(self, model, l):
        self.v, self.m, self.alpha, self.actif, self.p0 = None, 0.0, 1.0, False, 0
        self.handle = model.model.layers[l].register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        if self.v is None or not self.actif:
            return None
        h = sortie(out)
        v = self.v.to(device=h.device, dtype=torch.float32)
        hs = h[:, self.p0:].float()
        hs = hs - self.alpha * ((hs @ v) - self.m).unsqueeze(-1) * v
        h2 = torch.cat([h[:, :self.p0], hs.to(h.dtype)], 1)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    def remove(self):
        self.handle.remove()


class Capteur:
    """Capture, sur une tranche de positions, la sortie de chaque couche et l'entrée de la normalisation finale."""

    def __init__(self, model):
        self.tranche, self.couches, self.pre = None, {}, None
        self.handles = [l.register_forward_hook(self._hook(i)) for i, l in enumerate(model.model.layers)]
        self.handles.append(model.model.norm.register_forward_pre_hook(self._pre))

    def _hook(self, i):
        def f(mod, args, out):
            if self.tranche is not None:
                self.couches[i] = sortie(out)[:, self.tranche].float()
        return f

    def _pre(self, mod, args):
        if self.tranche is not None:
            self.pre = args[0]

    def remove(self):
        for h in self.handles:
            h.remove()


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


def contexte(g, enc, idx, res, info, loc, bras):
    lon = T1 + NPOS - 1
    ct = [res["temoin"]["tok"][i, :lon].tolist() for i in loc]
    conts = {"temoin": ct, "choc": [res["choc"]["tok"][i, :lon].tolist() for i in loc],
             "rejoue": [c[:T1] + [int(info["X"][i])] + c[T1 + 1:] for c, i in zip(ct, loc)]}
    return [enc[int(idx[i])] for i in loc] * len(bras), [x for a in bras for x in conts[a]]


@torch.no_grad()
def rejouer_S(g, idx, enc, res, info, bras=("temoin", "choc", "rejoue")):
    """Entropie S_q aux indices t₀ … t₀ + 31 (indice i : loi du jeton i) pour chaque bras, crochets actifs compris."""
    base, head = g.model.model, g.model.get_output_embeddings()
    P = len(idx)
    SR = {a: np.zeros((P, NPOS), np.float32) for a in bras}
    nb = 12
    for b0 in range(0, P, nb):
        loc = list(range(b0, min(P, b0 + nb)))
        ctxs, conts = contexte(g, enc, idx, res, info, loc, bras)
        n, Kp = len(ctxs), len(conts[0]) + 1
        ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        if g.proj is not None:                       # ablation à partir du jeton t₀
            g.proj.actif, g.proj.p0 = True, L - Kp + 1 + T1
        h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
        if g.proj is not None:
            g.proj.actif = False
        q0 = L - Kp + T1
        hk = h[:, q0:q0 + NPOS, :].reshape(n * NPOS, -1)
        flat = np.zeros(n * NPOS)
        for c0 in range(0, n * NPOS, 192):
            flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
        S = flat.reshape(n, NPOS)
        m = len(loc)
        for ai, a in enumerate(bras):
            SR[a][loc] = S[ai * m:(ai + 1) * m]
    return SR


def entropie_q(z):
    p = torch.softmax(z.double(), dim=-1)
    return (1.0 - (p ** Q).sum(-1)) / (Q - 1.0)


@torch.no_grad()
def flux(g, cap, enc, idx, res, info, l_star=None):
    """Passes rejouées témoin / rejoué avec états cachés. Renvoie ρ_l(k) = ‖δ_l‖ / ‖h_l(témoin)‖ (P, L, NPOS), le rapport de
    normes avant la normalisation finale, cos(δ_L, ∇S), la prédiction g·δ_L et l'écart d'entropie réel ; avec l_star, le
    second moment des δ et les moments du témoin à la couche l_star (k = 1–31)."""
    base, head, norm = g.model.model, g.model.get_output_embeddings(), g.model.model.norm
    nl, P = len(base.layers), len(idx)
    rho = np.zeros((P, nl, NPOS), np.float32)
    Rn, cosg, pred, dS = (np.zeros((P, NPOS), np.float32) for _ in range(4))
    acc = None
    verif = {"pre_vs_couche": 0.0}
    nb = 8
    for b0 in range(0, P, nb):
        loc = list(range(b0, min(P, b0 + nb)))
        ctxs, conts = contexte(g, enc, idx, res, info, loc, ("temoin", "rejoue"))
        n, Kp = len(ctxs), len(conts[0]) + 1
        m = n // 2
        ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        q0 = L - Kp + T1
        cap.tranche = slice(q0, q0 + NPOS)
        base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
        cap.tranche = None
        for l in range(nl):
            H = cap.couches[l].to(ids.device)
            d = H[m:] - H[:m]
            rho[loc, l] = (d.norm(dim=-1) / H[:m].norm(dim=-1).clamp(min=1e-6)).cpu().numpy()
        pre = cap.pre[:, q0:q0 + NPOS]
        verif["pre_vs_couche"] = max(verif["pre_vs_couche"], float((pre.float() - cap.couches[nl - 1].to(pre.device)).abs().max()))
        nt, nr = pre[:m].float().norm(dim=-1), pre[m:].float().norm(dim=-1)
        Rn[loc] = (nr / nt.clamp(min=1e-6)).cpu().numpy()
        dL = (pre[m:].float() - pre[:m].float())
        with torch.enable_grad():
            x = pre[:m].detach().clone().to(head.weight.device).requires_grad_(True)
            z = head(norm(x)).float()
            entropie_q(z).sum().backward()
            gr = x.grad.float()
        with torch.no_grad():
            dLd = dL.to(gr.device)
            cosg[loc] = ((dLd * gr).sum(-1) / (dLd.norm(dim=-1) * gr.norm(dim=-1)).clamp(min=1e-12)).cpu().numpy()
            pred[loc] = (dLd * gr).sum(-1).cpu().numpy()
            zt, zr = head(norm(pre[:m])).float(), head(norm(pre[m:])).float()
            dS[loc] = (entropie_q(zr) - entropie_q(zt)).float().cpu().numpy()
        if l_star is not None:
            H = cap.couches[l_star].to(ids.device)
            dl = (H[m:, 1:] - H[:m, 1:]).reshape(-1, H.shape[-1]).double()
            ht = H[:m, 1:].reshape(-1, H.shape[-1]).double()
            if acc is None:
                dd = H.shape[-1]
                acc = {"M2": torch.zeros(dd, dd, dtype=torch.float64, device=ids.device),
                       "S1": torch.zeros(dd, dtype=torch.float64, device=ids.device),
                       "S2": torch.zeros(dd, dd, dtype=torch.float64, device=ids.device), "n": 0,
                       "Dsum": torch.zeros(dd, dtype=torch.float64, device=ids.device)}
            acc["M2"] += dl.T @ dl
            acc["Dsum"] += dl.sum(0)
            acc["S1"] += ht.sum(0)
            acc["S2"] += ht.T @ ht
            acc["n"] += ht.shape[0]
        cap.couches, cap.pre = {}, None
    return {"rho": rho, "Rn": Rn, "cos": cosg, "pred": pred, "dS": dS, "acc": acc, "verif": verif}


@torch.no_grad()
def run_suite(g, enc, idx, res, info):
    """Témoin et choc régénérés à partir de t₀ pour les prompts idx : le début (jetons 0 … t₀ − 1) est repris de la génération
    sans ablation (l'ablation ne commence qu'à t₀), même bruit que A1c–A1u, même jeton choqué X."""
    P = len(idx)
    rc = {a: {"S": np.zeros((P, GEN_LEN), np.float32), "tok": np.zeros((P, GEN_LEN), np.int64)} for a in ("temoin", "choc")}
    ic = {"X": info["X"][idx].copy(), "egal": np.zeros(P, bool)}
    npb = max(1, g.budget // 2)
    for b0 in range(0, P, npb):
        loc = list(range(b0, min(P, b0 + npb)))
        rows_p = [int(idx[i]) for i in loc]
        n = len(loc)

        def force(s, z, nxt, n=n, loc=loc):
            if s == 0:                                                 # pas t₀ : le choc reçoit X
                for j, i in enumerate(loc):
                    x = int(ic["X"][i])
                    ic["egal"][i] = x == int(nxt[j])
                    nxt[n + j] = x
            return nxt

        pref = [enc[r] + res["temoin"]["tok"][r, :T1].tolist() for r in rows_p]
        S, tk = g.sample(pref * 2, rows_p * 2, T_MAIN[0], GEN_LEN - T1, T1, force)
        for ai, a in enumerate(("temoin", "choc")):
            sl = slice(ai * n, (ai + 1) * n)
            rc[a]["tok"][loc, :T1], rc[a]["S"][loc, :T1] = res["temoin"]["tok"][rows_p, :T1], res["temoin"]["S"][rows_p, :T1]
            rc[a]["tok"][loc, T1:], rc[a]["S"][loc, T1:] = tk[sl], S[sl]
    return rc, ic


@torch.no_grad()
def verif_crochet(g, proj, enc, idx, res, l):
    """Passe sur 4 textes du témoin, crochet actif à partir de t₀ : |⟨h, v⟩ − m| / ‖h‖ après t₀ et écart à la passe sans
    ablation avant t₀, à la sortie de la couche l."""
    lon = T1 + NPOS - 1
    seqs = [enc[int(r)] + res["temoin"]["tok"][int(r), :lon].tolist() for r in idx[:4]]
    ids, attn, L = g.left_pad(seqs)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    p0 = L - lon + T1
    hs = []
    hk = g.model.model.layers[l].register_forward_hook(lambda m_, a_, o_: hs.append(sortie(o_).float()))
    try:
        v_, act = proj.v, proj.actif
        proj.v = None
        g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
        proj.v, proj.actif, proj.p0 = v_, True, p0
        g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
        proj.actif = act
    finally:
        hk.remove()
    h0, h1 = hs
    vv = proj.v.to(h1.device)
    apres = float(((h1[:, p0:] @ vv - proj.m).abs() / h1[:, p0:].norm(dim=-1).clamp(min=1e-6)).max())
    avant = float((h1[:, :p0] - h0[:, :p0]).abs().max())
    return apres, avant

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
W = slice(MID[0], MID[1] + 1)


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1u2.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def effets_F(SR):
    Rl, Rr = SR["choc"] - SR["temoin"], SR["rejoue"] - SR["temoin"]
    return Rr[:, W].mean(1).astype(float), (Rl - Rr)[:, W].mean(1).astype(float)


for label, ids_m in MODELS:
    t_start = time.time()
    model = tok = g = cap = proj = None
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
        idx_all = np.arange(len(enc))
        t0g = time.time()
        res, info = run_main(g, enc, idx_all)
        print(f"[MESURE] {label} génération témoin + choc : {time.time() - t0g:.0f} s ; rang médian du choc "
              f"{np.median(info['rang']):.0f}, surprise médiane {np.median(info['surp']):.2f} nats", flush=True)
        cal, test = idx_all[idx_all % 2 == 0], idx_all[idx_all % 2 == 1]
        sub = lambda R, ix: {k: {kk: vv[ix] for kk, vv in v.items()} for k, v in R.items()}
        res_cal, info_cal = sub(res, cal), {k: v[cal] for k, v in info.items()}

        # Calibration : couche l*, v₁, moments du témoin à l* (comme A1u)
        cap = Capteur(model)
        Fc = flux(g, cap, enc, cal, res_cal, info_cal)
        nl = Fc["rho"].shape[1]
        rho_cal = Fc["rho"][:, :nl - 1, W].mean((0, 2))
        l_calc = int(np.argmax(rho_cal))
        l_star = A1U[label]["l_star"] if label in A1U else l_calc
        acc = flux(g, cap, enc, cal, res_cal, info_cal, l_star)["acc"]
        cap.remove()
        cap = None
        ev, evec = np.linalg.eigh(acc["M2"].cpu().numpy())
        v_calc = evec[:, -1]
        mu = acc["S1"].cpu().numpy() / acc["n"]
        C = acc["S2"].cpu().numpy() / acc["n"] - np.outer(mu, mu)
        v_1, src = v_calc, "recalculé"
        if label in V1_A1U_B64:
            v_a = np.frombuffer(base64.b64decode("".join(V1_A1U_B64[label])), dtype="<f4").astype(np.float64)
            if v_a.shape == v_calc.shape:
                v_1, src = v_a / np.linalg.norm(v_a), "celui d'A1u"
                ok_md5 = hashlib.md5(v_a.astype("<f4").tobytes()).hexdigest() == A1U[label]["v1_md5"]
                print(f"[ANCRAGE] {label} : couche l* recalculée {l_calc} (A1u : {A1U[label]['l_star']}) ; |cos(v₁ recalculé, v₁ "
                      f"d'A1u)| {abs(v_calc @ v_1):.4f} ; MD5 de v₁ embarqué conforme : {ok_md5}")
        print(f"[CONFIG] {label} : ablation à la couche {l_star}, v₁ {src}", flush=True)
        ec, ucv = np.linalg.eigh(C)
        U, lam = ucv[:, -N_PC:], np.clip(ec[-N_PC:], 0, None)
        rngA = np.random.default_rng(SEED + 13)                          # même tirage que A1u (20 premières)
        aleas = []
        for _ in range(N_ALEA):
            u = U @ (np.sqrt(lam) * rngA.normal(0, 1, N_PC))
            u -= (u @ v_1) * v_1
            aleas.append(u / np.linalg.norm(u))
        moy = lambda u: (float(mu @ u), float(math.sqrt(max(u @ C @ u, 1e-30))))
        m_v, sd_v = moy(v_1)
        ms = [moy(u) for u in aleas]
        z_al = [abs(m) / s for m, s in ms]
        print(f"[MOYENNE] {label} projection du témoin à l* : v₁ m = {m_v:+.3f}, écart-type {sd_v:.3f}, |m| / é.-t. "
              f"{abs(m_v) / sd_v:.2f} ; témoins aléatoires |m| / é.-t. médiane {np.median(z_al):.2f} [{min(z_al):.2f} ; "
              f"{max(z_al):.2f}] ; v₁ au-dessus de tous : {abs(m_v) / sd_v > max(z_al)}")

        proj = Projecteur(model, l_star)
        g.proj = proj
        conds = {"aucune": (None, 0.0, 1.0), "v1_moyenne": (v_1, m_v, 1.0), "v1_moitie": (v_1, m_v, 0.5),
                 "v1_zero": (v_1, 0.0, 1.0), **{f"alea_{j}": (u, m, 1.0) for j, (u, (m, _)) in enumerate(zip(aleas, ms))}}
        Ab, ta = {}, time.time()
        for ci_, (c, (vec, m, alpha)) in enumerate(conds.items()):
            proj.v = None if vec is None else torch.tensor(vec, dtype=torch.float32)
            proj.m, proj.alpha = m, alpha
            if c in ("v1_moyenne", "v1_zero"):
                apres, avant = verif_crochet(g, proj, enc, test, res, l_star)
                print(f"[VÉRIF] {label} crochet {c} : |⟨h, v⟩ − m| / ‖h‖ après t₀, max {apres:.1e} ; écart avant t₀ (doit "
                      f"être 0) : {avant:.1e}", flush=True)
                if apres > 1e-2 or avant > 1e-3:
                    raise RuntimeError("crochet d'ablation défectueux : résultats non interprétables")
            rc, ic = run_suite(g, enc, test, res, info)
            SRc = rejouer_S(g, test, enc, rc, ic)
            Dc, Ic = effets_F(SRc)
            chg = float(np.mean((rc["temoin"]["tok"][:, T1:] != res["temoin"]["tok"][test, T1:]).any(1)))
            Ab[c] = {"D": Dc, "I": Ic, "S_temoin": float(SRc["temoin"][:, W].mean()), "egal": float(ic["egal"].mean()),
                     "suites_changees": chg}
            if ci_ == 0:
                est = (time.time() - ta) * len(conds) / 60
                print(f"[VÉRIF] {label} sans ablation, régénéré à partir de t₀ : suites du témoin différentes de la génération "
                      f"complète {chg:.1%} (bascules numériques possibles en fp16)")
                print(f"[PROGRESSION] {label} ablations : une condition en {time.time() - ta:.0f} s ; estimation ~{est:.0f} min "
                      f"pour {len(conds)} conditions", flush=True)
                if est / 60 > MAX_HOURS:
                    raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
        proj.v, proj.alpha = None, 1.0
        print(f"[MESURE] {label} ablations : {time.time() - ta:.0f} s", flush=True)

        Pt = len(test)
        rngB = np.random.default_rng(SEED + 3)
        bt_t = [rngB.integers(0, Pt, Pt) for _ in range(BOOT_N)]
        F_auc = regle_F(Ab["aucune"]["D"], Ab["aucune"]["I"], bt_t, ok)
        al = [c for c in conds if c.startswith("alea_")]
        S0 = Ab["aucune"]["S_temoin"]
        dg = {c: Ab[c]["S_temoin"] - S0 for c in conds}
        I_al, d_al = [float(Ab[c]["I"].mean()) for c in al], [dg[c] for c in al]
        D_al = [float(Ab[c]["D"].mean()) for c in al]
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} : {Pt} prompts de test, couche {l_star}")
        print(f"[F] {label} sans ablation : D = {F_auc['D']:+.4f} ; I = {F_auc['I']:+.4f} IC95 [{F_auc['I_ci'][0]:+.4f} ; "
              f"{F_auc['I_ci'][1]:+.4f}] → {F_auc['verdict']}"
              + (f" (A1u, mêmes prompts : I = {A1U[label]['I_test']:+.4f})" if label in A1U else ""))
        for c in ("v1_moyenne", "v1_moitie", "v1_zero"):
            print(f"[DÉGÂT] {label} {c:<10} : entropie du témoin {Ab[c]['S_temoin']:.3f} (sans ablation {S0:.3f}, d = {dg[c]:+.3f}) ; "
                  f"suites du témoin changées {Ab[c]['suites_changees']:.0%} ; D = {Ab[c]['D'].mean():+.4f} ; I = {Ab[c]['I'].mean():+.4f}")
        print(f"[DÉGÂT] {label} témoins aléatoires : d médian {np.median(d_al):+.3f} [{min(d_al):+.3f} ; {max(d_al):+.3f}] ; "
              f"suites changées médiane {np.median([Ab[c]['suites_changees'] for c in al]):.0%} ; D médian {np.median(D_al):+.4f} ; "
              f"I médian {np.median(I_al):+.4f} [{min(I_al):+.4f} ; {max(I_al):+.4f}]")
        v4b = regle_v4b(Ab["aucune"]["I"], Ab["v1_moyenne"]["I"], I_al, dg["v1_moyenne"], d_al, bt_t, F_auc["verdict"], ok)
        rang = int(sum(x < Ab["v1_moyenne"]["I"].mean() for x in I_al))
        aj = ajuste(float(Ab["v1_moyenne"]["I"].mean()), dg["v1_moyenne"], I_al, d_al)
        rm = regle_v4b(Ab["aucune"]["I"], Ab["v1_moitie"]["I"], I_al, dg["v1_moitie"], d_al, bt_t, F_auc["verdict"], ok)
        repli = v4b["verdict"] == "NON TESTABLE (DÉGÂT HORS DES TÉMOINS)"
        rz = regle_v4b(Ab["aucune"]["I"], Ab["v1_zero"]["I"], I_al, dg["v1_zero"], d_al, bt_t, F_auc["verdict"], ok)
        print(f"[V4b] {label} v₁ par la moyenne : r = 1 − I(v₁) / I(aucune) = {v4b['r']:+.2f} IC95 [{v4b['ci'][0]:+.2f} ; "
              f"{v4b['ci'][1]:+.2f}] ; témoins avec un I plus bas : {rang} sur {len(I_al)} ; dégât dans la plage des témoins : "
              f"{v4b['apparie']} → {v4b['verdict']}")
        print(f"[V4b] {label} ajusté au dégât (droite I ~ d sur les témoins, pente {aj['pente']:+.3f}) : résidu de v₁ "
              f"{aj['residu']:+.4f}, soit {aj['z']:+.1f} écarts-types ; {aj['rang']} témoins sur {len(I_al)} ont un résidu plus bas"
              + (" ; EXTRAPOLATION (v₁ hors de la plage des dégâts)" if aj["extrapolation"] else ""))
        print(f"[V4b] {label} repli, v₁ retiré à moitié : r = {rm['r']:+.2f} IC95 [{rm['ci'][0]:+.2f} ; {rm['ci'][1]:+.2f}] ; "
              f"dégât dans la plage : {rm['apparie']} → {rm['verdict']}" + (" (verdict retenu)" if repli else " (non utilisé)"))
        print(f"[V4b] {label} v₁ par zéro (descriptif) : r = {rz['r']:+.2f} IC95 [{rz['ci'][0]:+.2f} ; {rz['ci'][1]:+.2f}] ; "
              f"dégât dans la plage : {rz['apparie']} → {rz['verdict']}")
        ALL[label] = {"l_star": l_star, "l_recalcule": l_calc, "v1_source": src, "m_v1": m_v, "sd_v1": sd_v,
                      "z_alea": z_al, "F_test": F_auc, "V4b": v4b, "V4b_repli": rm, "repli_utilise": repli, "ajuste": aj,
                      "V4b_zero": rz, "rang": rang,
                      "ablations": {c: {"D": float(x["D"].mean()), "I": float(x["I"].mean()), "S_temoin": x["S_temoin"],
                                        "degat": dg[c], "suites_changees": x["suites_changees"]} for c, x in Ab.items()},
                      "minutes": (time.time() - t_start) / 60}
        np.savez_compressed(os.path.join(OUT, f"{label}_a1u2.npz"), test=test,
                            **{f"D_{c}": x["D"] for c, x in Ab.items()}, **{f"I_{c}": x["I"] for c, x in Ab.items()})
        print(f"[DURÉE] {label} : {ALL[label]['minutes']:.1f} min", flush=True)
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}", flush=True)
        ALL[label] = {"erreur": f"{type(e).__name__}: {str(e)[:300]}"}
    for o in (cap, proj):
        if o is not None:
            o.remove()
    del model, tok, g, cap, proj
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ecrire_resume()

# %%
# ---------- Synthèse ----------
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1u2 — la direction v₁ porte-t-elle l'amplification, à dégât égal ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "V4b" in ALL[m]]
these = "NON TESTABLE (modèle manquant)"
if labs:
    x = ALL[labs[0]]
    these = (synthese(x["V4b_repli"]["verdict"], ok) + " [repli, v₁ retiré à moitié]") if x["repli_utilise"] else synthese(x["V4b"]["verdict"], ok)
for m in labs:
    print(f"[SYNTHÈSE] {m:<18} : V4b {ALL[m]['V4b']['verdict']} ; repli {ALL[m]['V4b_repli']['verdict']} "
          f"({'utilisé' if ALL[m]['repli_utilise'] else 'non utilisé'}) ; v₁ par zéro (descriptif) {ALL[m]['V4b_zero']['verdict']}")
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « à dégât égal, v₁ porte une part spécifique de l'amplification » : {these} (prédiction : "
      f"SOUTENUE 35 %, RÉFUTÉE 50 %, NON TESTABLE 15 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

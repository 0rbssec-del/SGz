# %% [markdown]
# # SGZ A1u — Le mécanisme de l'amplification : divergence, norme, gradient d'entropie, direction unique ?
#
# **Point de départ.** La trajectoire amplifie un choc (24 verdicts sur 24, A1c–A1m) ; quatre explications simples sont
# tombées (valeur, redistribution entre têtes, routage, attention verrouillée). Deux modèles extérieurs, interrogés par
# Erick (28 oct.), proposent des pistes : Argon 4 (une direction du flux résiduel trouvée par SVD des écarts choc − témoin,
# un « gain de Lyapunov » > 1, un test rapide de la norme avant la normalisation finale) ; Qwen3 (neurones MLP, SAE).
# A1u teste ce qui est mesurable, sans la partie « q gouverne le système » (q est notre mesure, pas un réglage du modèle).
#
# **Dispositif (celui de A1c et A1t).** Qwen3-1.7B-Base et SmolLM3-3B-Base, 768 prompts, T = 0,6, 96 jetons, choc fort
# (rang 1000–5000) au pas t₀ = 64. Passes rejouées : témoin (suite du témoin) et rejoué (même suite, jeton choqué à t₀) :
# seuls les états cachés diffèrent, jamais les jetons, donc l'écart δ_l(k) = h_l(rejoué) − h_l(témoin) mesure la seule
# propagation du choc. k = position après t₀ (k = 1 : le jeton choqué lui-même).
#
# ## Mesures et critères (fixés avant toute exécution ; bootstrap sur les prompts, B = 1000)
#
# - **V1, Lyapunov empirique** (les deux modèles) : pour chaque couche l, ρ_l(k) = moyenne sur les prompts de
#   ‖δ_l(k)‖ / ‖h_l(témoin, k)‖ ; λ_l = pente des moindres carrés de ln ρ_l(k) sur k = 2–31. Verdict sur la dernière couche
#   (entrée de la normalisation finale) : DIVERGENT si IC95(λ) > 0 ; STATIONNAIRE si IC95 ⊂ [−0,005 ; +0,005] ; CONTRACTANT si
#   IC95 < 0 ; sinon INDÉTERMINÉ. Descriptif : λ_l et son verdict pour chaque couche.
# - **V2, norme avant la normalisation finale** (test rapide d'Argon) : R = moyenne sur k = 2–31 de ‖h_L(rejoué)‖ /
#   ‖h_L(témoin)‖ : HAUSSE PERSISTANTE si IC95 > 1,02 ; ABSENTE si IC95 ⊂ [0,98 ; 1,02] ; BAISSE si IC95 < 0,98 ; sinon
#   INDÉTERMINÉE. (Argon : hausse persistante → mécanisme par la normalisation et les MLP ; norme stable → par l'attention.)
# - **V3, gradient d'entropie** : g = ∇_h S_q (q = 1,45) de la loi de sortie, pris en h_L(témoin) ; cos(δ_L, g) moyen sur
#   k = 2–31 : POUSSE VERS L'ENTROPIE si IC95 > 0 ; CONTRE si IC95 < 0 ; sinon NON ALIGNÉ. Descriptif : prédiction au premier
#   ordre g·δ contre l'écart d'entropie réel (corrélation, part expliquée).
# - **V4, une direction unique ?** (Qwen3-1.7B-Base seulement) : couche l* = celle où ρ_l est le plus grand sur k = 2–31,
#   parmi les couches 1 à L − 1, choisie sur les prompts pairs (calibration). v₁ = premier vecteur propre du second moment
#   des δ_{l*}(k), k = 1–31, sur la calibration (descriptif : part de variance du premier axe, le critère 0,5 d'Argon).
#   Test sur les prompts impairs (384) : génération et passes rejouées refaites avec v₁ retiré du flux résiduel à la sortie
#   de la couche l*, à toutes les positions, même jeton choqué que sans ablation. Effet de la trajectoire I (règle F de A1c).
#   Témoins : 20 directions aléatoires tirées dans le sous-espace des 64 premiers axes principaux des activations du témoin
#   à l*, orthogonales à v₁ (rapport de variance projetée imprimé). Réduction r = 1 − I(v₁) / I(aucune) :
#   MÉCANISME PRINCIPAL si borne basse de IC95(r) ≥ 0,5 et I(v₁) inférieur aux 20 témoins ; CONTRIBUTION SPÉCIFIQUE si borne
#   basse > 0 et I(v₁) inférieur aux 20 témoins ; NON SPÉCIFIQUE si borne basse > 0 sans cela ; AUGMENTE si borne haute < 0 ;
#   sinon PAS D'EFFET. NON TESTABLE si I(aucune) n'est pas AMPLIFICATION (règle F).
# - **Synthèses.** Thèse d'Argon « l'écart diverge dans le flux résiduel » (V1, deux modèles) : SOUTENUE si DIVERGENT deux
#   fois ; RÉFUTÉE si CONTRACTANT deux fois ; sinon INDÉTERMINÉE. Thèse d'Argon « une direction unique porte
#   l'amplification » (V4) : SOUTENUE si MÉCANISME PRINCIPAL ; PARTIELLE si CONTRIBUTION SPÉCIFIQUE ; RÉFUTÉE si NON
#   SPÉCIFIQUE, PAS D'EFFET ou AUGMENTE ; sinon INDÉTERMINÉE.
# - Témoins positifs synthétiques pour chaque règle ; vérifications : crochet d'ablation (projection résiduelle ≈ 0, aucun
#   écart sans direction), dernier état caché = normalisation finale de l'entrée capturée, δ nul avant t₀.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1 CONTRACTANT 75 %, DIVERGENT 10 %. V2 ABSENTE 40 %, HAUSSE PERSISTANTE 30 %, BAISSE 20 %. V3 POUSSE VERS L'ENTROPIE 65 %.
# V4 MÉCANISME PRINCIPAL 15 %, CONTRIBUTION SPÉCIFIQUE 30 %, NON SPÉCIFIQUE 30 %, PAS D'EFFET 20 %. Thèses : divergence
# RÉFUTÉE 70 % ; direction unique SOUTENUE 15 %, RÉFUTÉE 55 %.
#
# ## Limites connues
#
# - Une seule couche d'ablation (l*) et une seule direction ; un mécanisme réparti sur plusieurs couches ou directions
#   apparaîtrait comme NON SPÉCIFIQUE ou PAS D'EFFET.
# - Retirer v₁ à toutes les positions touche aussi le témoin ; I est un contraste à l'intérieur de chaque condition.
# - Lyapunov « empirique » : pente d'une moyenne sur 30 pas, pas un exposant asymptotique.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~70 min pour Qwen3
# (dont ~55 min d'ablations), ~10 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`,
# `[V1]` à `[V4]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1u" if os.path.isdir("/kaggle/working") else "sgz_a1u_out")
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
STAT_LAMBDA, MARGE_NORME = 0.005, 0.02
N_ALEA, N_PC = 20, 64
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1v
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
ABLATION = {"Qwen3-1.7B-Base"}                    # V4 sur ce modèle seulement
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1 = 16, 48, 12
    RANK_LO, RANK_HI = 20, 80
    MID = (2, 12)
    N_ALEA, N_PC = 3, 8
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-aleatoire-bis", ["smoke-qwen3-random"])]
    ABLATION = {"smoke-qwen3-aleatoire"}
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
assert T1 + MID[1] <= GEN_LEN - 1
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T1} fenêtre k={MID[0]}–{MID[1]} ; ablation sur "
      f"{sorted(ABLATION)} ({N_ALEA} directions aléatoires dans {N_PC} axes) ; modèles={[m[0] for m in MODELS]}")
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


KS = np.arange(MID[0], MID[1] + 1)


def pente(rho_k):
    """Pente des moindres carrés de ln(ρ) sur k ; rho_k : (nk,) moyennes sur les prompts."""
    y = np.log(np.maximum(rho_k, 1e-12))
    x = KS - KS.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def regle_lyap(rho, boots, interpretable=True):
    """rho : (P, nk) rapports ‖δ‖ / ‖h‖ par prompt sur k = 2–31."""
    lam = pente(rho.mean(0))
    c = ci([pente(rho[ix].mean(0)) for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > 0:
        v = "DIVERGENT"
    elif c[0] >= -STAT_LAMBDA and c[1] <= STAT_LAMBDA:
        v = "STATIONNAIRE"
    elif c[1] < 0:
        v = "CONTRACTANT"
    else:
        v = "INDÉTERMINÉ"
    return {"lambda": lam, "ci": c, "verdict": v}


def regle_norme(R, boots, interpretable=True):
    r = float(R.mean())
    c = ci([R[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > 1 + MARGE_NORME:
        v = "HAUSSE PERSISTANTE"
    elif c[0] >= 1 - MARGE_NORME and c[1] <= 1 + MARGE_NORME:
        v = "ABSENTE"
    elif c[1] < 1 - MARGE_NORME:
        v = "BAISSE"
    else:
        v = "INDÉTERMINÉE"
    return {"R": r, "ci": c, "verdict": v}


def regle_cos(C, boots, interpretable=True):
    m = float(C.mean())
    c = ci([C[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > 0:
        v = "POUSSE VERS L'ENTROPIE"
    elif c[1] < 0:
        v = "CONTRE"
    else:
        v = "NON ALIGNÉ"
    return {"cos": m, "ci": c, "verdict": v}


def regle_ablation(I0, Iv, I_alea, boots, verdict_F0, interpretable=True):
    """I0, Iv : effets de la trajectoire par prompt sans ablation et avec v₁ retiré ; I_alea : moyennes des témoins."""
    r = float(1 - Iv.mean() / I0.mean())
    c = ci([1 - Iv[ix].mean() / I0[ix].mean() for ix in boots])
    sous_tous = bool(Iv.mean() < min(I_alea)) if len(I_alea) else False
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif verdict_F0 != "AMPLIFICATION":
        v = f"NON TESTABLE (sans ablation : {verdict_F0})"
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
    return {"r": r, "ci": c, "sous_tous_les_temoins": sous_tous, "verdict": v}


def synthese_div(v1, interpretable=True):
    vs = list(v1.values())
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if vs and all(v == "DIVERGENT" for v in vs):
        return "SOUTENUE"
    if vs and all(v == "CONTRACTANT" for v in vs):
        return "RÉFUTÉE"
    return "INDÉTERMINÉE"


def synthese_dir(v4, interpretable=True):
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if v4 == "MÉCANISME PRINCIPAL":
        return "SOUTENUE"
    if v4 == "CONTRIBUTION SPÉCIFIQUE":
        return "PARTIELLE"
    if v4 in ("NON SPÉCIFIQUE", "PAS D'EFFET", "AUGMENTE"):
        return "RÉFUTÉE"
    return "INDÉTERMINÉE"


# Témoins positifs synthétiques
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]


def rho_synth(lam):
    a = rng_pc.lognormal(np.log(0.3), 0.3, Pp)[:, None]
    return a * np.exp(lam * KS)[None, :] * rng_pc.lognormal(0, 0.1, (Pp, len(KS)))


I0_pc = 0.05 + rng_pc.normal(0, 0.1, Pp)
bruit = lambda: (lambda x: x - x.mean())(rng_pc.normal(0, 0.02, Pp))   # centré : ne déplace pas la moyenne
checks = [("divergent", regle_lyap(rho_synth(0.03), bt)["verdict"], "DIVERGENT"),
          ("contractant", regle_lyap(rho_synth(-0.05), bt)["verdict"], "CONTRACTANT"),
          ("stationnaire", regle_lyap(rho_synth(0.0), bt)["verdict"], "STATIONNAIRE"),
          ("norme en hausse", regle_norme(1.10 + rng_pc.normal(0, 0.05, Pp), bt)["verdict"], "HAUSSE PERSISTANTE"),
          ("norme stable", regle_norme(1.0 + rng_pc.normal(0, 0.05, Pp), bt)["verdict"], "ABSENTE"),
          ("norme en baisse", regle_norme(0.90 + rng_pc.normal(0, 0.05, Pp), bt)["verdict"], "BAISSE"),
          ("aligné", regle_cos(0.05 + rng_pc.normal(0, 0.1, Pp), bt)["verdict"], "POUSSE VERS L'ENTROPIE"),
          ("non aligné", regle_cos((lambda x: x - x.mean())(rng_pc.normal(0, 0.1, Pp)), bt)["verdict"], "NON ALIGNÉ"),
          ("mécanisme principal", regle_ablation(I0_pc, 0.2 * I0_pc + bruit(), [0.05, 0.048, 0.051], bt, "AMPLIFICATION")["verdict"],
           "MÉCANISME PRINCIPAL"),
          ("contribution", regle_ablation(I0_pc, 0.7 * I0_pc + bruit(), [0.05, 0.048, 0.051], bt, "AMPLIFICATION")["verdict"],
           "CONTRIBUTION SPÉCIFIQUE"),
          ("non spécifique", regle_ablation(I0_pc, 0.7 * I0_pc + bruit(), [0.03, 0.048, 0.051], bt, "AMPLIFICATION")["verdict"],
           "NON SPÉCIFIQUE"),
          ("pas d'effet", regle_ablation(I0_pc, I0_pc + bruit(), [0.05, 0.048, 0.051], bt, "AMPLIFICATION")["verdict"], "PAS D'EFFET"),
          ("sans amplification", regle_ablation(I0_pc, 0.2 * I0_pc, [0.05], bt, "ÉQUIVALENT")["verdict"],
           "NON TESTABLE (sans ablation : ÉQUIVALENT)"),
          ("divergence réfutée", synthese_div({"a": "CONTRACTANT", "b": "CONTRACTANT"}), "RÉFUTÉE"),
          ("direction partielle", synthese_dir("CONTRIBUTION SPÉCIFIQUE"), "PARTIELLE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<22} : {got:<24} (attendu {att})")
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
# ---------- Choc à t₀, crochets (ablation, capture des états cachés), passes rejouées ----------
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
    """Retire la composante selon v (vecteur unitaire) de la sortie d'une couche, à toutes les positions."""

    def __init__(self, model, l):
        self.v = None
        self.handle = model.model.layers[l].register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        if self.v is None:
            return None
        h = sortie(out)
        v = self.v.to(device=h.device, dtype=h.dtype)
        h2 = h - (h @ v).unsqueeze(-1) * v
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
        h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
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

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
W = slice(MID[0], MID[1] + 1)


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1u.json"), "w") as fh:
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
        SR0 = rejouer_S(g, idx_all, enc, res, info)
        d = np.abs(SR0["temoin"] - res["temoin"]["S"][:, T1:T1 + NPOS])
        print(f"[VÉRIF] {label} témoin rejoué contre généré : écart moyen {d.mean():.1e}, max {d.max():.1e}", flush=True)
        if d.mean() > 0.1:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
        P = len(enc)
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        D0, I0 = effets_F(SR0)
        F0 = regle_F(D0, I0, boots, ok)
        print(f"[F] {label} sans ablation, {P} prompts : D = {F0['D']:+.4f} ; I = {F0['I']:+.4f} IC95 [{F0['I_ci'][0]:+.4f} ; "
              f"{F0['I_ci'][1]:+.4f}] → {F0['verdict']} (reproduction de A1c)")

        cap = Capteur(model)
        tf = time.time()
        Fx = flux(g, cap, enc, idx_all, res, info)
        nl = Fx["rho"].shape[1]
        print(f"[VÉRIF] {label} : entrée de la normalisation finale contre sortie de la dernière couche : écart max "
              f"{Fx['verif']['pre_vs_couche']:.1e} ; écart avant le choc (k = 0) : ρ max {Fx['rho'][:, :, 0].max():.1e} ; "
              f"passes avec états cachés : {time.time() - tf:.0f} s", flush=True)
        if Fx["verif"]["pre_vs_couche"] > 1e-2 or Fx["rho"][:, :, 0].max() > 1e-2:
            raise RuntimeError("capture des états cachés désalignée : résultats non interprétables")
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts, {nl} couches")
        lyap = [regle_lyap(Fx["rho"][:, l, W].astype(float), boots, ok) for l in range(nl)]
        for l in range(nl):
            r_ = Fx["rho"][:, l]
            print(f"[LYAPUNOV] {label} couche {l:>2} : ρ(k=1) {r_[:, 1].mean():.3f} ; ρ(k=2) {r_[:, 2].mean():.3f} ; ρ(k=31) "
                  f"{r_[:, MID[1]].mean():.3f} ; λ = {lyap[l]['lambda']:+.4f} IC95 [{lyap[l]['ci'][0]:+.4f} ; {lyap[l]['ci'][1]:+.4f}] "
                  f"→ {lyap[l]['verdict']}")
        v1 = lyap[-1]
        nb_div = sum(x["verdict"] == "DIVERGENT" for x in lyap)
        print(f"[V1] {label} Lyapunov empirique, dernière couche : λ = {v1['lambda']:+.4f} IC95 [{v1['ci'][0]:+.4f} ; "
              f"{v1['ci'][1]:+.4f}] → {v1['verdict']} ; couches DIVERGENT : {nb_div} sur {nl}")
        v2 = regle_norme(Fx["Rn"][:, W].mean(1).astype(float), boots, ok)
        print(f"[V2] {label} norme avant la normalisation finale, rejoué / témoin (k = 2–31) : {v2['R']:.4f} IC95 "
              f"[{v2['ci'][0]:.4f} ; {v2['ci'][1]:.4f}] → {v2['verdict']} ; à k = 1 : {Fx['Rn'][:, 1].mean():.4f}")
        v3 = regle_cos(Fx["cos"][:, W].mean(1).astype(float), boots, ok)
        pr, ds = Fx["pred"][:, W].ravel(), Fx["dS"][:, W].ravel()
        corr = float(np.corrcoef(pr, ds)[0, 1])
        part = float(pr.mean() / ds.mean()) if ds.mean() != 0 else float("nan")
        print(f"[V3] {label} cos(δ, ∇S) moyen (k = 2–31) : {v3['cos']:+.4f} IC95 [{v3['ci'][0]:+.4f} ; {v3['ci'][1]:+.4f}] → "
              f"{v3['verdict']} ; prédiction au premier ordre g·δ contre écart réel : corrélation {corr:+.3f}, moyenne prédite / "
              f"réelle {part:+.2f} ; à k = 1 : cos {Fx['cos'][:, 1].mean():+.4f}")
        a = {"F0": F0, "V1": v1, "V1_couches": lyap, "V2": v2, "V3": v3, "premier_ordre": {"corr": corr, "part": part}}

        if label in ABLATION:
            cal, test = idx_all[idx_all % 2 == 0], idx_all[idx_all % 2 == 1]
            rho_cal = Fx["rho"][cal][:, :nl - 1, W].mean((0, 2))
            l_star = int(np.argmax(rho_cal))
            print(f"[V4] {label} couche d'ablation l* = {l_star} (ρ moyen sur la calibration {rho_cal[l_star]:.3f} ; "
                  f"choisie parmi les couches 0 à {nl - 2})", flush=True)
            sub = lambda R, ix: {k: {kk: vv[ix] for kk, vv in v.items()} for k, v in R.items()}
            res_cal, info_cal = sub(res, cal), {k: v[cal] for k, v in info.items()}
            acc = flux(g, cap, enc, cal, res_cal, info_cal, l_star)["acc"]
            M2 = acc["M2"].cpu().numpy()
            ev, evec = np.linalg.eigh(M2)
            v_1 = evec[:, -1]
            part1 = float(ev[-1] / ev.sum())
            dmoy = acc["Dsum"].cpu().numpy()
            cos_moy = float(abs(v_1 @ dmoy) / (np.linalg.norm(dmoy) + 1e-12))
            mu = acc["S1"].cpu().numpy() / acc["n"]
            C = acc["S2"].cpu().numpy() / acc["n"] - np.outer(mu, mu)
            ec, ucv = np.linalg.eigh(C)
            U, lam = ucv[:, -N_PC:], np.clip(ec[-N_PC:], 0, None)
            var_v1 = float(v_1 @ C @ v_1)
            rngA = np.random.default_rng(SEED + 13)
            aleas = []
            for _ in range(N_ALEA):
                u = U @ (np.sqrt(lam) * rngA.normal(0, 1, N_PC))
                u -= (u @ v_1) * v_1
                aleas.append(u / np.linalg.norm(u))
            ratios = [float(u @ C @ u) / var_v1 for u in aleas]
            print(f"[V4] {label} v₁ : part de variance du premier axe des δ {part1:.3f} (critère 0,5 d'Argon, descriptif) ; "
                  f"|cos(v₁, δ moyen)| {cos_moy:.3f} ; variance projetée des témoins aléatoires / celle de v₁ : médiane "
                  f"{np.median(ratios):.2f} [{min(ratios):.2f} ; {max(ratios):.2f}]", flush=True)
            cap.remove()
            cap = None
            proj = Projecteur(model, l_star)
            conds = {"aucune": None, "v1": v_1, **{f"alea_{j}": u for j, u in enumerate(aleas)}}
            X_test = info["X"][test]
            Ab, ta = {}, time.time()
            for ci_, (c, vec) in enumerate(conds.items()):
                proj.v = None if vec is None else torch.tensor(vec, dtype=torch.float32)
                if c == "v1":
                    ids_v, attn_v, _ = g.left_pad([enc[int(r)] for r in test[:4]])
                    hs = []
                    hk = model.model.layers[l_star].register_forward_hook(lambda m_, a_, o_: hs.append(sortie(o_).float()))
                    with torch.no_grad():
                        model.model(input_ids=ids_v, attention_mask=attn_v, use_cache=False)
                        hk.remove()
                        hv = hs[0]
                        resid = float(((hv @ proj.v.to(hv.device)).abs() / hv.norm(dim=-1).clamp(min=1e-6)).max())
                    print(f"[VÉRIF] {label} crochet d'ablation : |⟨h, v₁⟩| / ‖h‖ après projection, max {resid:.1e}", flush=True)
                    if resid > 1e-2:
                        raise RuntimeError("crochet d'ablation défectueux : résultats non interprétables")
                rc, ic = run_main(g, enc, test, X_fixe=X_test)
                SRc = rejouer_S(g, test, enc, rc, ic)
                Dc, Ic = effets_F(SRc)
                Ab[c] = {"D": Dc, "I": Ic, "S_temoin": float(SRc["temoin"][:, W].mean()), "egal": float(ic["egal"].mean())}
                if ci_ == 0:
                    est = (time.time() - ta) * len(conds) / 60
                    print(f"[PROGRESSION] {label} ablations : une condition en {time.time() - ta:.0f} s ; estimation ~{est:.0f} min "
                          f"pour {len(conds)} conditions", flush=True)
                    if est / 60 > MAX_HOURS:
                        raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
            proj.v = None
            Pt = len(test)
            bt_t = [rngB.integers(0, Pt, Pt) for _ in range(BOOT_N)]
            F_auc = regle_F(Ab["aucune"]["D"], Ab["aucune"]["I"], bt_t, ok)
            I_al = [float(Ab[c]["I"].mean()) for c in conds if c.startswith("alea_")]
            D_al = [float(Ab[c]["D"].mean()) for c in conds if c.startswith("alea_")]
            v4 = regle_ablation(Ab["aucune"]["I"], Ab["v1"]["I"], I_al, bt_t, F_auc["verdict"], ok)
            print(f"[V4] {label} test ({Pt} prompts) sans ablation : D = {F_auc['D']:+.4f}, I = {F_auc['I']:+.4f} → {F_auc['verdict']}")
            print(f"[V4] {label} v₁ retiré : D = {Ab['v1']['D'].mean():+.4f}, I = {Ab['v1']['I'].mean():+.4f} ; témoins aléatoires : "
                  f"I médian {np.median(I_al):+.4f} [{min(I_al):+.4f} ; {max(I_al):+.4f}], D médian {np.median(D_al):+.4f}")
            print(f"[V4] {label} dégât sur le témoin (entropie moyenne k = 2–31) : aucune {Ab['aucune']['S_temoin']:.3f} ; v₁ "
                  f"{Ab['v1']['S_temoin']:.3f} ; aléatoires médiane {np.median([Ab[c]['S_temoin'] for c in conds if c.startswith('alea_')]):.3f} ; "
                  f"choc égal au jeton du témoin : {Ab['v1']['egal']:.1%}")
            print(f"[V4] {label} réduction de I par v₁ : r = {v4['r']:+.2f} IC95 [{v4['ci'][0]:+.2f} ; {v4['ci'][1]:+.2f}] ; "
                  f"sous les {len(I_al)} témoins : {v4['sous_tous_les_temoins']} → {v4['verdict']}")
            a.update({"l_star": l_star, "part_premier_axe": part1, "cos_v1_dmoy": cos_moy, "ratios_variance": ratios,
                      "F_test": F_auc, "V4": v4, "ablations": {c: {"D": float(x["D"].mean()), "I": float(x["I"].mean()),
                                                                  "S_temoin": x["S_temoin"]} for c, x in Ab.items()}})
            np.save(os.path.join(OUT, f"{label}_v1.npy"), v_1)
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1u.npz"), rho=Fx["rho"], Rn=Fx["Rn"], cos=Fx["cos"],
                            pred=Fx["pred"], dS=Fx["dS"], X=info["X"], **{f"S0_{b}": SR0[b] for b in SR0})
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min", flush=True)
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1u — le mécanisme de l'amplification")
v1s = {m: ALL[m]["V1"]["verdict"] for m in ALL if isinstance(ALL[m], dict) and "V1" in ALL[m]}
for m in v1s:
    x = ALL[m]
    print(f"[SYNTHÈSE] {m:<18} : V1 {x['V1']['verdict']:<14} ; V2 {x['V2']['verdict']:<18} ; V3 {x['V3']['verdict']:<22}"
          + (f" ; V4 {x['V4']['verdict']}" if "V4" in x else ""))
t1 = "NON TESTABLE (modèle manquant)" if len(v1s) < len(MODELS) else synthese_div(v1s, ok)
v4s = [ALL[m]["V4"]["verdict"] for m in ALL if isinstance(ALL[m], dict) and "V4" in ALL[m]]
t2 = synthese_dir(v4s[0], ok) if v4s else "NON TESTABLE (ablation manquante)"
ALL["these_divergence"], ALL["these_direction"] = t1, t2
print(f"[SYNTHÈSE] thèse d'Argon « l'écart diverge dans le flux résiduel » : {t1} (prédiction RÉFUTÉE 70 %)")
print(f"[SYNTHÈSE] thèse d'Argon « une direction unique porte l'amplification » : {t2} (prédiction SOUTENUE 15 %, RÉFUTÉE 55 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

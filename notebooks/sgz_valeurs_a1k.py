# %% [markdown]
# # SGZ A1k — Le jeton anormal « crie »-t-il plus fort ? Norme ‖W_O v‖ de sa valeur, par tête et par couche
#
# **But.** Sur 10 configurations (A1c–A1i), la contribution N = ‖Σ_h α_h W_O^h v_h‖ du jeton choqué à la sortie de
# l'attention est toujours plus forte que celle du jeton témoin (×1,44 à ×2,9), alors que son poids d'attention A varie
# selon la famille (×0,72 chez OLMo à ×1,45 chez SmolLM3). Hypothèse : le jeton anormal produit un **vecteur de valeur
# plus grand**, indépendamment de l'attention qu'il reçoit. A1k mesure directement, au pas t₀ et pour chaque couche et
# chaque tête : ‖W_O^h v_h‖ (ce que le jeton enverrait à attention égale), ‖v_h‖ (avant W_O) et ‖x‖ (l'état qui entre
# dans l'attention : après la norme d'entrée pour Qwen, SmolLM3, Granite et Mistral ; flux résiduel brut pour OLMo-2).
#
# **Dispositif.** Cinq modèles qui couvrent l'éventail de A : Qwen3-1.7B-Base (A ×1,21), SmolLM3-3B-Base (×1,45),
# granite-3.3-8b-base (×1,22), OLMo-2-1124-7B (×0,72), Mistral-7B-v0.3 (×0,94). Mêmes 768 prompts, même graine, même choc
# (rang 1000–5000) au pas t₀ = 64, T = 0,6 ; génération arrêtée à 96 jetons (la fenêtre k = 8–31 suffit), donc mêmes
# trajectoires que A1c–A1i sur ces 96 jetons. Passes rejouées en « eager » avec les bras témoin, fort et doux.
#
# ## Critères (fixés avant toute exécution)
#
# - **V** (mesure principale) = moyenne sur les couches de la norme quadratique moyenne sur les têtes de ‖W_O^h v_h(t₀)‖,
#   par prompt. ρ_V = V(choc fort) / V(témoin), bootstrap apparié (B = 1000) ; ÉQUIVALENT si IC95 ⊂ [0,8 ; 1,2] ;
#   SUPPRESSION si IC95 < 1 ; SAILLANCE si IC95 > 1 ; sinon INDÉTERMINÉ (même règle que A et N).
# - **Hypothèse « valeur »** : SOUTENUE si ρ_V = SAILLANCE pour au moins 4 modèles sur 5 ; RÉFUTÉE si ρ_V = ÉQUIVALENT ou
#   SUPPRESSION pour au moins 3 modèles sur 5 ; sinon INDÉTERMINÉE.
# - **Part de N expliquée par la valeur** (par modèle) : log ρ_V / log ρ_N, IC95 bootstrap. L'ESSENTIEL si borne basse > 0,5 ;
#   MINORITAIRE si borne haute < 0,5 ; sinon PARTAGÉE. Le reste vient du poids d'attention et de l'alignement des têtes.
# - **Ancrages** : ρ_A et ρ_N (k = 8–31) comparés aux valeurs publiées par A1d, A1f et A1i ; pour Granite, OLMo et
#   Mistral, trajectoires identiques à A1i (MD5 des 96 premiers jetons). Un écart est signalé, sans invalider ρ_V.
# - Secondaires, sans verdict : ρ pour ‖v‖ et pour ‖x‖ ; ρ_V par couche ; ρ_V du choc doux ; corrélation de Spearman, par
#   prompt, entre log(V fort / V témoin) et la surprise du jeton forcé.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Hypothèse « valeur » SOUTENUE 55 %, RÉFUTÉE 20 % ; part de N expliquée L'ESSENTIEL pour au moins 3 modèles : 45 % ;
# choc doux ρ_V ÉQUIVALENT pour au moins 4 modèles : 70 %.
#
# ## Limites connues
#
# - OLMo-2 normalise la sortie de l'attention après coup (post-norm) : pour lui, ‖W_O v‖ est mesuré avant cette norme.
# - V est une moyenne non pondérée par l'attention : il décrit ce que le jeton enverrait, pas ce qui est effectivement lu.
# - Une seule température ; corrélation avec la surprise descriptive seulement.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : ~10 à 20 min par modèle, ~1 h 15 en tout [estimation].
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[ANCRAGE]`, `[VÉRIF]`, `[V]`, `[PART]`, `[HYPOTHÈSE]`.

# %%
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1k" if os.path.isdir("/kaggle/working") else "sgz_a1k_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 768, 96, 64                # 96 jetons : fenêtre k = 8–31 complète
RANK_LO, RANK_HI = 1000, 5000
W_MAIN, MID_D = (8, 31), (2, 31)
MID_F, MARGE_REL = (2, 31), 0.3
WINS = [(2, 7), (8, 31)]
MARGE = 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                   # même graine que A1c : mêmes trajectoires
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))    # par modèle
ARMS_R = ["temoin", "fort", "doux"]
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base", "ibm-granite/granite-3.3-8b-base",
                    "allenai/OLMo-2-1124-7B", "mistralai/Mistral-7B-v0.3"]   # l'un après l'autre
REF = {"Qwen3-1.7B-Base": (1.214, 2.884, "A1d"), "SmolLM3-3B-Base": (1.451, 1.971, "A1f"),
       "granite-3.3-8b-base": (1.221, 2.327, "A1i"), "OLMo-2-1124-7B": (0.723, 1.689, "A1i"),
       "Mistral-7B-v0.3": (0.941, 2.298, "A1i")}                      # ρ_A, ρ_N du choc fort à T = 0,6
A1I_MD5 = {"granite-3.3-8b-base": ("95d0d30cca783bea3f4266816a4c1ca4", "2e93c307f9a4e1c558835f539074e02d"),
           "OLMo-2-1124-7B": ("d7fc138bbbb63873f338a1a8032e7ab9", "9863a2b7722802b76fe9b8ef743cf1c0"),
           "Mistral-7B-v0.3": ("bbd88551d0fa2e5078909cac5998f720", "e7d285ae8f9217949158523f1c00cf04")}   # 96 premiers jetons
try:                                                             # jeton Hugging Face facultatif (secret Kaggle)
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("HF_TOKEN", UserSecretsClient().get_secret("HF_TOKEN"))
    print("[CONFIG] jeton Hugging Face trouvé dans les secrets Kaggle")
except Exception:
    pass
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 16, 72, 24
    RANK_LO, RANK_HI = 20, 80
    W_MAIN, MID_D = (4, 12), (2, 12)
    MID_F = (2, 12)
    WINS = [(2, 3), (4, 12), (13, 47)]
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]
K = GEN_LEN - T0 - 1
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t0={T0} fenêtre k={W_MAIN[0]}–{W_MAIN[1]} "
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
print(f"[PROMPTS] {len(PROMPTS)} prompts ({min(64, len(PROMPTS))} écrits à la main, {max(0, len(PROMPTS) - 64)} gabarits), "
      f"identiques à A1c")

# %%
# ---------- Mesures, règles de décision et témoin positif ----------
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


def ratio_verdict(xs, xt, boots, interpretable=True):
    """ρ = moyenne(bras choqué) / moyenne(témoin), bootstrap apparié sur les prompts."""
    r = float(xs.mean() / xt.mean())
    c = ci([xs[ix].mean() / xt[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
        v = "ÉQUIVALENT"
    elif c[1] < 1:
        v = "SUPPRESSION"
    elif c[0] > 1:
        v = "SAILLANCE"
    else:
        v = "INDÉTERMINÉ"
    return {"rho": r, "ci": c, "verdict": v}


def md5(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a).astype(np.int64)).tobytes()).hexdigest()


# Témoin positif : rapports connus, bruit multiplicatif par prompt (log-normal, σ = 0,5)
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
base_pc = rng_pc.lognormal(np.log(0.01), 0.5, Pp)
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
noisy = lambda f: f * base_pc * rng_pc.lognormal(0, 0.3, Pp)
checks = [
    ("suppression (×0,5)", ratio_verdict(noisy(0.5), base_pc, bt)["verdict"], "SUPPRESSION"),
    ("équivalent (×1,0)", ratio_verdict(noisy(1.0), base_pc, bt)["verdict"], "ÉQUIVALENT"),
    ("saillance (×1,6)", ratio_verdict(noisy(1.6), base_pc, bt)["verdict"], "SAILLANCE"),
]
for name, got, exp in checks:
    print(f"[TÉMOIN POSITIF] {name:<20} : {got:<12} (attendu {exp})")


def trajectoire(Rlib, Rrej, boots, interpretable=True):
    """Volet F (A1c) : effet direct D (bras rejoué) et effet de la trajectoire I = libre − rejoué."""
    D = win(Rrej, MID_F)
    cD = ci([D[ix].mean() for ix in boots])
    I = win(Rlib, MID_F) - D
    cI = ci([I[ix].mean() for ix in boots])
    m = MARGE_REL * abs(D.mean())
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cD[0] <= 0 <= cD[1]:
        v = "NON TESTABLE (effet direct absent)"
    elif cI[0] >= -m and cI[1] <= m:
        v = "ÉQUIVALENT"
    elif cI[1] < 0:
        v = "COMPENSATION"
    elif cI[0] > 0:
        v = "AMPLIFICATION"
    else:
        v = "INDÉTERMINÉ"
    return {"D": float(D.mean()), "ci_D": cD, "I": float(I.mean()), "ci_I": cI, "marge": float(m),
            "rapport": float(I.mean() / D.mean()) if D.mean() != 0 else float("nan"), "verdict": v}


K_ = GEN_LEN - T0 - 1
kk = np.arange(1, K_ + 1)


def synth_pair(factor, amp=1.0, decay=0.8, Pn=Pp):
    direct = amp * decay ** kk
    rej = direct[None, :] + rng_pc.normal(0, 0.05, (Pn, 1)) + rng_pc.normal(0, 0.1, (Pn, K_))
    lib = factor * direct[None, :] + rng_pc.normal(0, 0.2, (Pn, 1)) + rng_pc.normal(0, 0.3, (Pn, K_))
    return lib, rej


for name, fac, amp, exp in (("F compensation", 0.4, 1.0, "COMPENSATION"), ("F équivalent", 1.0, 1.0, "ÉQUIVALENT"),
                            ("F amplification", 1.6, 1.0, "AMPLIFICATION"),
                            ("F effet direct nul", 1.0, 0.0, "NON TESTABLE (effet direct absent)")):
    got = trajectoire(*synth_pair(fac, amp), bt)["verdict"]
    checks.append((name, got, exp))
    print(f"[TÉMOIN POSITIF] {name:<20} : {got:<12} (attendu {exp})")


def part_valeur(vf, vt, nf, nt, boots, interpretable=True):
    """Part de log ρ_N portée par log ρ_V (bootstrap apparié) ; L'ESSENTIEL si borne basse > 0,5."""
    f = lambda ix: np.log(vf[ix].mean() / vt[ix].mean()) / np.log(nf[ix].mean() / nt[ix].mean())
    cN = ci([nf[ix].mean() / nt[ix].mean() for ix in boots])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = float(f(np.arange(len(vf))))
        c = ci([f(ix) for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif cN[0] <= 1 <= cN[1]:
        v = "NON TESTABLE (ρ_N non distinct de 1)"
    elif c[0] > 0.5:
        v = "L'ESSENTIEL"
    elif c[1] < 0.5:
        v = "MINORITAIRE"
    else:
        v = "PARTAGÉE"
    return {"part": r, "ci": c, "verdict": v}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


vt_pc = base_pc * rng_pc.lognormal(0, 0.2, Pp)
for name, fv, fa, exp in (("part valeur essentielle", 2.0, 1.0, "L'ESSENTIEL"), ("part valeur minoritaire", 1.05, 2.4, "MINORITAIRE")):
    vf_pc = fv * vt_pc * rng_pc.lognormal(0, 0.2, Pp)
    nt_pc = vt_pc * rng_pc.lognormal(0, 0.2, Pp)
    nf_pc = fa * vf_pc / vt_pc * nt_pc * rng_pc.lognormal(0, 0.2, Pp)
    got = part_valeur(vf_pc, vt_pc, nf_pc, nt_pc, bt)["verdict"]
    checks.append((name, got, exp))
    print(f"[TÉMOIN POSITIF] {name:<20} : {got:<12} (attendu {exp})")
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
    elif max([float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b\b", model_id.lower())] or [0]) >= 6 and N_GPU >= 2:
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
# ---------- Génération (identique à A1c) ----------
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


@torch.no_grad()
def run_main(g, enc):
    """Témoin et choc fort empilés, comme A1c ; on note aussi le jeton du choc doux (le plus probable autre que le témoin)."""
    P = len(enc)
    forced_rank = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, P)
    res = {}
    npb = max(1, g.budget // 2)
    n_lots = len(T_MAIN) * math.ceil(P / npb)
    for T in T_MAIN:
        t0 = time.time()
        for a in ("temoin", "fort"):
            res[("S", T, a)] = np.zeros((P, GEN_LEN), np.float32)
            res[("tok", T, a)] = np.zeros((P, GEN_LEN), np.int64)
        soft = np.zeros(P, np.int64)
        rang = np.zeros(P, np.int64)
        surp = np.zeros(P, np.float32)
        surp_d = np.zeros(P, np.float32)
        for b0 in range(0, P, npb):
            tl = time.time()
            rows_p = list(range(b0, min(P, b0 + npb)))
            n = len(rows_p)

            def force(s, z, nxt, n=n, rows_p=rows_p):
                if s < T0:
                    nxt[n:] = nxt[:n]                              # appariement exact avant le choc
                elif s == T0:
                    zz = z[n:].clone()
                    zz[:, g.forbid] = -float("inf")
                    if g.V > g.n_tok:
                        zz[:, g.n_tok:] = -float("inf")
                    order = torch.argsort(zz, dim=-1, descending=True)
                    lp = torch.log_softmax(z[n:].float(), dim=-1)           # distribution non tempérée p₁
                    for j, r in enumerate(rows_p):
                        rk = int(min(forced_rank[r], order.shape[1] - 2))
                        if int(order[j, rk]) == int(nxt[j]):     # jamais le jeton du témoin
                            rk += 1
                        nxt[n + j] = order[j, rk]
                        rang[r] = rk
                        soft[r] = int(order[j, 0]) if int(order[j, 0]) != int(nxt[j]) else int(order[j, 1])
                        surp[r] = float(-lp[j, nxt[n + j]])
                        surp_d[r] = float(-lp[j, soft[r]])
                return nxt

            S, tk = g.sample([enc[r] for r in rows_p] * 2, rows_p * 2, T, GEN_LEN, 0, force)
            for ai, a in enumerate(("temoin", "fort")):
                sl = slice(ai * n, (ai + 1) * n)
                res[("S", T, a)][rows_p], res[("tok", T, a)][rows_p] = S[sl], tk[sl]
            g.progress(time.time() - tl, n_lots)
        res[("soft", T)], res[("surp", T)], res[("surp_doux", T)] = soft, surp, surp_d
        ref, fo = res[("tok", T, "temoin")], res[("tok", T, "fort")]
        same = np.array_equal(ref[:, :T0], fo[:, :T0])
        chg = float(np.mean(ref[:, T0] != fo[:, T0]))
        dsoft = float(np.mean(soft != ref[:, T0]))
        print(f"[VÉRIF] {g.label} T={T} : identique avant t0 : {same} ; jeton changé à t0 : fort {chg:.0%}, doux {dsoft:.0%} ; "
              f"rang médian du choc fort {int(np.median(rang))}", flush=True)
        if not same or chg < 0.99 or dsoft < 0.99:
            raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
        if not SMOKE_TEST and g.label in A1I_MD5:
            ok_t, ok_f = md5(ref) == A1I_MD5[g.label][0], md5(fo) == A1I_MD5[g.label][1]
            print(f"[ANCRAGE] {g.label} T={T} : trajectoires identiques à A1i (MD5, 96 jetons) : témoin {ok_t}, choc fort {ok_f}"
                  + ("" if ok_t and ok_f else " (écart numérique possible ; l'analyse reste valide sur ces trajectoires)"), flush=True)
        print(f"[MESURE] {g.label} T={T} : {time.time() - t0:.0f} s", flush=True)
    return res

# %%
# ---------- Passes rejouées avec lecture de l'attention vers le jeton du pas t₀ ----------
class AttnProbe:
    """Crochets sur chaque couche d'attention : poids α(requête t₀ + k → clé t₀) par tête, et contribution
    ‖Σ_h α_h W_O^h v_h(t₀)‖ du jeton t₀ à la sortie de l'attention."""

    def __init__(self, model):
        self.layers = model.model.layers
        self.nL = len(self.layers)
        self.active, self.check, self.err = False, False, 0.0
        self.kidx = self.qidx = None
        self.v, self.A, self.N, self.Ah, self.U, self.Vn, self.X = ([None] * self.nL for _ in range(7))
        self.handles = []
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(partial(self.pre, l), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(partial(self.post, l)))

    def pre(self, l, mod, args, kwargs):
        if not self.active:
            return
        hs = kwargs.get("hidden_states", args[0] if args else None)
        b = torch.arange(hs.shape[0], device=hs.device)
        x = hs[b, self.kidx.to(hs.device)]
        self.v[l] = mod.v_proj(x).float()                                      # valeurs du jeton t₀ (après la norme)
        self.X[l] = x.float().norm(dim=-1).cpu()                               # ‖x‖ : état entrant dans l'attention

    def post(self, l, mod, args, output):
        if not self.active:
            return
        w = output[1]
        if w is None:
            raise RuntimeError("poids d'attention absents : l'implémentation « eager » est requise")
        B, H = w.shape[0], w.shape[1]
        k = self.kidx.to(w.device)
        q = self.qidx.to(w.device)
        wk = w[torch.arange(B, device=w.device), :, :, k]                       # (B, H, L) : vers la clé t₀
        al = wk.gather(2, q[:, None, :].expand(B, H, q.shape[1])).float()       # (B, H, K)
        hd = mod.head_dim
        v = self.v[l].view(B, -1, hd).repeat_interleave(mod.num_key_value_groups, dim=1)   # (B, H, hd)
        Wo = mod.o_proj.weight.float().view(-1, H, hd)                          # (d, H, hd)
        u = torch.einsum("bhe,dhe->bhd", v, Wo)                                 # contribution par tête
        self.U[l] = u.norm(dim=-1).cpu()                                        # ‖W_O^h v_h‖ (B, H)
        self.Vn[l] = v.norm(dim=-1).cpu()                                       # ‖v_h‖ avant W_O (B, H)
        G = torch.einsum("bhd,bgd->bhg", u, u)
        N = torch.einsum("bhk,bhg,bgk->bk", al, G, al).clamp(min=0).sqrt()
        if self.check:
            c = torch.einsum("bhk,bhd->bkd", al, u).norm(dim=-1)
            self.err = max(self.err, float((c - N).abs().max() / (c.abs().max() + 1e-12)))
        self.A[l], self.N[l] = al.mean(1).cpu(), N.cpu()
        self.Ah[l] = al[:, :, W_MAIN[0] - 1:W_MAIN[1]].mean(-1).cpu()          # (B, H)


@torch.no_grad()
def replay_attn(g, probe, ctxs, conts):
    base, head = g.model.model, g.model.get_output_embeddings()
    n, Kp = len(ctxs), len(conts[0]) + 1
    ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    off = [L - len(c) - len(s) for c, s in zip(ctxs, conts)]
    kidx = torch.tensor([o + len(c) + T0 for o, c in zip(off, ctxs)])
    qidx = torch.tensor([[o + len(c) + T0 + k - 1 for k in range(1, K + 1)] for o, c in zip(off, ctxs)])
    ids_cpu = ids.cpu()
    for j in range(n):                                                         # alignement : la clé est bien le jeton t₀
        if int(ids_cpu[j, kidx[j]]) != int(conts[j][T0]) or int(ids_cpu[j, qidx[j, 0]]) != int(conts[j][T0]):
            raise RuntimeError("index de la clé t₀ mal aligné : résultats non interprétables")
    probe.kidx, probe.qidx, probe.active = kidx, qidx, True
    h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    probe.active = False
    starts = [L - len(s) - 1 for s in conts]
    idx = torch.tensor([[st + i for i in range(Kp)] for st in starts], device=h.device)
    hk = h[torch.arange(n, device=h.device)[:, None], idx].reshape(n * Kp, -1)
    flat = np.zeros(n * Kp, np.float64)
    for c0 in range(0, n * Kp, 192):
        z = head(hk[c0:c0 + 192].to(head.weight.device)).float()
        cfg = g.model.config
        if getattr(cfg, "logits_scaling", None):                         # Granite : logits divisés à la sortie
            z = z / cfg.logits_scaling
        if getattr(cfg, "final_logit_softcapping", None):                # Gemma : plafonnement des logits
            z = torch.tanh(z / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
        flat[c0:c0 + 192] = entropies_from_logits(z, [Q])[Q].cpu().numpy()
    A = torch.stack(probe.A, 1).cpu().numpy()                                  # (n, couches, K)
    N = torch.stack(probe.N, 1).cpu().numpy()
    Ah = torch.stack(probe.Ah, 1).cpu().numpy()                                # (n, couches, têtes)
    nrm = {k: torch.stack(getattr(probe, k), 1).cpu().numpy() for k in ("U", "Vn", "X")}   # (n, couches[, têtes])
    return flat.reshape(n, Kp).astype(np.float32), A, N, Ah, nrm


@torch.no_grad()
def run_replay(g, probe, enc, res):
    P = len(enc)
    nb = 8
    for T in T_MAIN:
        t0 = time.time()
        ref, fo, soft = res[("tok", T, "temoin")], res[("tok", T, "fort")], res[("soft", T)]
        for a in ARMS_R:
            res[("Srej", T, a)] = np.zeros((P, GEN_LEN), np.float32)
            res[("A", T, a)], res[("N", T, a)] = np.zeros((P, K), np.float32), np.zeros((P, K), np.float32)
            res[("Alw", T, a)] = np.zeros((P, probe.nL, len(WINS)), np.float32)
            res[("Nlw", T, a)] = np.zeros((P, probe.nL, len(WINS)), np.float32)
            res[("Ah", T, a)] = None
            res[("U", T, a)] = res[("Vn", T, a)] = res[("X", T, a)] = None
        for b0 in range(0, P, nb):
            rows = list(range(b0, min(P, b0 + nb)))
            ct = [ref[r, :GEN_LEN - 1].tolist() for r in rows]
            cf = [c[:T0] + [int(fo[r, T0])] + c[T0 + 1:] for c, r in zip(ct, rows)]
            cd = [c[:T0] + [int(soft[r])] + c[T0 + 1:] for c, r in zip(ct, rows)]
            probe.check = b0 == 0
            S, A, N, Ah, nrm = replay_attn(g, probe, [enc[r] for r in rows] * 3, ct + cf + cd)
            m = len(rows)
            for ai, a in enumerate(ARMS_R):
                sl = slice(ai * m, (ai + 1) * m)
                res[("Srej", T, a)][rows] = S[sl]
                res[("A", T, a)][rows], res[("N", T, a)][rows] = A[sl].mean(1), N[sl].mean(1)
                for wi, w in enumerate(WINS):
                    res[("Alw", T, a)][rows, :, wi] = A[sl][:, :, w[0] - 1:w[1]].mean(-1)
                    res[("Nlw", T, a)][rows, :, wi] = N[sl][:, :, w[0] - 1:w[1]].mean(-1)
                if res[("Ah", T, a)] is None:
                    res[("Ah", T, a)] = np.zeros((P,) + Ah.shape[1:], np.float32)
                res[("Ah", T, a)][rows] = Ah[sl]
                for k_, x_ in nrm.items():
                    if res[(k_, T, a)] is None:
                        res[(k_, T, a)] = np.zeros((P,) + x_.shape[1:], np.float32)
                    res[(k_, T, a)][rows] = x_[sl]
        d_all = np.abs(res[("Srej", T, "temoin")] - res[("S", T, "temoin")])
        d1 = np.abs(res[("Srej", T, "fort")][:, T0 + 1] - res[("S", T, "fort")][:, T0 + 1])
        print(f"[VÉRIF] {g.label} T={T} rejoué (eager) contre généré : témoin écart moyen {d_all.mean():.1e}, max {d_all.max():.1e} ; "
              f"choc à k=1 {d1.mean():.1e} ; calcul de N (Gram contre somme directe) écart relatif {probe.err:.1e}", flush=True)
        if d_all.mean() > 0.1 or d1.mean() > 0.1 or probe.err > 1e-3:
            raise RuntimeError("passe rejouée ou calcul de N incohérent : résultats non interprétables")
        print(f"[MESURE] {g.label} T={T} passes rejouées : {time.time() - t0:.0f} s", flush=True)
    return res

# %%
# ---------- Analyse ----------
TRANCHES = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 95)]


def tranches(X):
    return [float(X[:, x - 1:y].mean()) for x, y in TRANCHES if y <= X.shape[1]]


def analyse(res, interpretable):
    ok = interpretable and POSITIVE_CONTROL_OK
    rng = np.random.default_rng(SEED + 3)
    P = res[("S", T_MAIN[0], "temoin")].shape[0]
    boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]
    out = {"A": {}, "N": {}, "D": {}, "couches": {}, "tetes": {}, "courbes": {}}
    wi = WINS.index(W_MAIN)
    for T in T_MAIN:
        for meas in ("A", "N"):
            xt = win(res[(meas, T, "temoin")], W_MAIN)
            out[meas][T] = {a: ratio_verdict(win(res[(meas, T, a)], W_MAIN), xt, boots, ok) for a in ("fort", "doux")}
            for a in ARMS_R:
                out["courbes"][f"{meas} {T}|{a}"] = tranches(res[(meas, T, a)])
            lw = res[(meas[0] + "lw", T, "temoin")][:, :, wi].mean(0)
            out["couches"][f"{meas} {T}"] = {a: (res[(meas[0] + "lw", T, a)][:, :, wi].mean(0) / lw).tolist() for a in ("fort", "doux")}
        D = {}
        for a in ("fort", "doux"):
            R = (res[("Srej", T, a)] - res[("Srej", T, "temoin")])[:, T0 + 1:].astype(float)
            d = win(R, MID_D)
            D[a] = {"D": float(d.mean()), "ci": ci([d[ix].mean() for ix in boots]), "courbe": tranches(R)}
        out["D"][T] = D
        ht = res[("Ah", T, "temoin")].mean(0)
        rf, rd = res[("Ah", T, "fort")].mean(0) / ht, res[("Ah", T, "doux")].mean(0) / ht
        order = np.argsort(rf.ravel())
        pick = list(order[:5]) + list(order[-5:])
        out["tetes"][T] = [{"couche": int(i // rf.shape[1]), "tete": int(i % rf.shape[1]), "rho_fort": float(rf.ravel()[i]),
                            "rho_doux": float(rd.ravel()[i]), "attention_temoin": float(ht.ravel()[i])} for i in pick]
        out["tetes"][f"{T}_part"] = {"fort<0.8": float(np.mean(rf < 0.8)), "fort>1.25": float(np.mean(rf > 1.25)),
                                     "doux<0.8": float(np.mean(rd < 0.8)), "doux>1.25": float(np.mean(rd > 1.25))}
    def supp(meas, T):
        return out[meas][T]["fort"]["verdict"] == "SUPPRESSION" and out[meas][T]["doux"]["verdict"] == "ÉQUIVALENT"
    nonsupp = lambda meas, T: out[meas][T]["fort"]["verdict"] in ("ÉQUIVALENT", "SAILLANCE")
    if not ok:
        these = "NON INTERPRÉTABLE" if not interpretable else "NON TESTABLE (témoin positif raté)"
    elif any(all(supp(m, T) for T in T_MAIN) for m in ("A", "N")):
        these = "SOUTENUE"
    elif all(nonsupp(m, T) for m in ("A", "N") for T in T_MAIN):
        these = "RÉFUTÉE"
    else:
        these = "INDÉTERMINÉE"
    out["these_suppression"] = these
    out["F"], out["courbes_F"] = {}, {}
    for T in T_MAIN:
        Rlib = (res[("S", T, "fort")] - res[("S", T, "temoin")])[:, T0 + 1:].astype(float)
        Rrej = (res[("Srej", T, "fort")] - res[("Srej", T, "temoin")])[:, T0 + 1:].astype(float)
        out["F"][T] = trajectoire(Rlib, Rrej, boots, ok)
        out["courbes_F"][f"{T}|libre"], out["courbes_F"][f"{T}|rejoué"] = tranches(Rlib), tranches(Rrej)
        out["courbes_F"][f"{T}|trajectoire"] = tranches(Rlib - Rrej)
    vF = [out["F"][T]["verdict"] for T in T_MAIN]
    if not ok:
        tF = these
    elif all(x == "COMPENSATION" for x in vF):
        tF = "SOUTENUE"
    elif all(x in ("ÉQUIVALENT", "AMPLIFICATION") for x in vF):
        tF = "RÉFUTÉE"
    else:
        tF = "INDÉTERMINÉE"
    out["these_trajectoire"] = tF
    if not ok:
        out["replication"] = these
    elif tF == "RÉFUTÉE" and these == "RÉFUTÉE":
        out["replication"] = "RÉPLIQUÉE"
    elif "SOUTENUE" in (tF, these):
        out["replication"] = "CONTREDITE"
    else:
        out["replication"] = "PARTIELLE"
    out["these"] = out["replication"]
    return out

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ALL = {}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1k.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "modeles": ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)

for label, ids in MODELS:
    t_start = time.time()
    model = tok = g = probe = res = None
    for mid in ids:
        print(f"[CHARGEMENT] {label} : téléchargement / chargement de {mid}…", flush=True)
        try:
            model, tok, mode = load_model(mid)
            print(f"[CHARGEMENT] {label} : {mid} en {mode}")
            break
        except Exception as e:
            print(f"[CHARGEMENT] échec {mid} : {type(e).__name__}: {str(e)[:300]}")
            model = None
    if model is None:
        ALL[label] = {"these": "NON TESTABLE (modèle non chargé)"}
        continue
    try:
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        g = Gen(model, tok, label)
        enc = [tok.encode(p)[:12] if SMOKE_TEST else tok.encode(p) for p in PROMPTS]
        print(f"[CONFIG] attention pendant la génération : {getattr(model.config, '_attn_implementation', '?')}")
        try:
            res = run_main(g, enc)
        except FloatingPointError as e:
            print(f"[VÉRIF] {label} : {e} en fp16 → rechargement en fp32 (même protocole)", flush=True)
            del g
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model, tok, mode = load_model(ids[0], fp32=True)
            if getattr(tok, "pad_token_id", None) is None:
                tok.pad_token = tok.eos_token
            print(f"[CHARGEMENT] {label} : {ids[0]} en {mode}", flush=True)
            g = Gen(model, tok, label)
            res = run_main(g, enc)
        print(f"[CONFIG] attention pendant les passes rejouées : {set_attention(model, 'eager')}")
        probe = AttnProbe(model)
        res = run_replay(g, probe, enc, res)
        for hnd in probe.handles:
            hnd.remove()
        a = analyse(res, interpretable)
        ok = interpretable and POSITIVE_CONTROL_OK
        T = T_MAIN[0]
        rngV = np.random.default_rng(SEED + 13)
        bootsV = [rngV.integers(0, len(enc), len(enc)) for _ in range(BOOT_N)]
        rms = lambda k_, arm: np.sqrt((res[(k_, T, arm)].astype(np.float64) ** 2).mean(-1))   # (P, couches)
        lay = {arm: rms("U", arm) for arm in ARMS_R}
        Vp = {arm: lay[arm].mean(1) for arm in ARMS_R}                                        # V par prompt
        m_l = lay["temoin"].mean(0)
        Ve = {arm: (lay[arm] / m_l).mean(1) for arm in ARMS_R}                                # couches à poids égal
        Vv = {arm: rms("Vn", arm).mean(1) for arm in ARMS_R}
        Xp = {arm: res[("X", T, arm)].astype(np.float64).mean(1) for arm in ARMS_R}
        Np = {arm: win(res[("N", T, arm)], W_MAIN) for arm in ARMS_R}
        a["V"] = {arm: ratio_verdict(Vp[arm], Vp["temoin"], bootsV, ok) for arm in ("fort", "doux")}
        a["V_poids_egal"] = {arm: ratio_verdict(Ve[arm], Ve["temoin"], bootsV, ok) for arm in ("fort", "doux")}
        a["v_avant_WO"] = {arm: ratio_verdict(Vv[arm], Vv["temoin"], bootsV, ok) for arm in ("fort", "doux")}
        a["x_entree"] = {arm: ratio_verdict(Xp[arm], Xp["temoin"], bootsV, ok) for arm in ("fort", "doux")}
        a["part_valeur"] = {arm: part_valeur(Vp[arm], Vp["temoin"], Np[arm], Np["temoin"], bootsV, ok) for arm in ("fort", "doux")}
        a["V_couches"] = {arm: (lay[arm].mean(0) / m_l).tolist() for arm in ("fort", "doux")}
        lr = np.log(Vp["fort"] / Vp["temoin"])
        a["spearman_surprise"] = spearman(lr, res[("surp", T)])
        a["surprise_mediane"] = {"fort": float(np.median(res[("surp", T)])), "doux": float(np.median(res[("surp_doux", T)]))}
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        ref = REF.get(label)
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T})")
        for meas, nom in (("A", "attention"), ("N", "contribution ‖αWv‖")):
            r = a[meas][T]["fort"]
            print(f"[ANCRAGE] {label} {meas} ({nom}, k={W_MAIN[0]}–{W_MAIN[1]}) choc fort : ρ = {r['rho']:.3f} "
                  f"IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}] {r['verdict']}"
                  + (f" ; référence {ref[2]} : {ref[0 if meas == 'A' else 1]:.3f}" if ref and not SMOKE_TEST else ""))
        for key, nom in (("V", "‖W_O v‖, mesure principale"), ("V_poids_egal", "‖W_O v‖, couches à poids égal"),
                         ("v_avant_WO", "‖v‖ avant W_O"), ("x_entree", "‖x‖ entrant dans l'attention")):
            for arm in ("fort", "doux"):
                r = a[key][arm]
                print(f"[V] {label} {nom}, choc {arm:<4} : {r['verdict']:<12} ρ = {r['rho']:.3f} "
                      f"IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}]")
        for arm in ("fort", "doux"):
            r = a["part_valeur"][arm]
            print(f"[PART] {label} choc {arm:<4} : log ρ_V / log ρ_N = {r['part']:+.2f} IC95 [{r['ci'][0]:+.2f} ; {r['ci'][1]:+.2f}] "
                  f"→ part de N portée par la valeur : {r['verdict']}")
        for arm in ("fort", "doux"):
            print(f"[COUCHES] {label} ρ_V par couche, choc {arm:<4} : " + " ".join(f"{v:.2f}" for v in a["V_couches"][arm]))
        print(f"[CORR] {label} surprise médiane du jeton : fort {a['surprise_mediane']['fort']:.1f} nats, doux "
              f"{a['surprise_mediane']['doux']:.1f} nats ; Spearman(log V fort/témoin, surprise du choc fort) = "
              f"{a['spearman_surprise']:+.2f}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a1k.npz"),
                            **{f"{k}_{arm}": res[(k, T, arm)] for k in ("S", "tok") for arm in ("temoin", "fort")},
                            soft=res[("soft", T)], surp=res[("surp", T)], surp_doux=res[("surp_doux", T)],
                            **{f"{k}_{arm}": res[(k, T, arm)].astype(np.float16 if k in ("Ah", "U", "Vn") else np.float32)
                               for k in ("Srej", "A", "N", "Ah", "U", "Vn", "X") for arm in ARMS_R})
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min")
        ecrire_resume()
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}")
        ALL[label] = {"these": f"NON TESTABLE ({type(e).__name__})"}
    del model, g, probe, res
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if not SMOKE_TEST:                                            # libère le disque avant le modèle suivant
        shutil.rmtree(os.path.join(HF_CACHE, "models--" + ids[0].replace("/", "--")), ignore_errors=True)

print(f"\n{'=' * 100}")
ok = interpretable and POSITIVE_CONTROL_OK
vs = {lab: a["V"]["fort"]["verdict"] for lab, a in ALL.items() if "V" in a}
n_s = sum(v == "SAILLANCE" for v in vs.values())
n_r = sum(v in ("ÉQUIVALENT", "SUPPRESSION") for v in vs.values())
if not ok:
    hyp = "NON INTERPRÉTABLE" if not interpretable else "NON TESTABLE (témoin positif raté)"
elif len(vs) < 5:
    hyp = f"NON TESTABLE ({len(vs)} modèles mesurés sur 5)"
elif n_s >= 4:
    hyp = "SOUTENUE"
elif n_r >= 3:
    hyp = "RÉFUTÉE"
else:
    hyp = "INDÉTERMINÉE"
for lab, a in ALL.items():
    if "V" in a:
        print(f"[SYNTHÈSE] {lab} : ρ_A {a['A'][T_MAIN[0]]['fort']['rho']:.2f}, ρ_N {a['N'][T_MAIN[0]]['fort']['rho']:.2f}, "
              f"ρ_V {a['V']['fort']['rho']:.2f} ({a['V']['fort']['verdict']}), part de N portée par la valeur "
              f"{a['part_valeur']['fort']['part']:+.2f} ({a['part_valeur']['fort']['verdict']}) ; choc doux ρ_V "
              f"{a['V']['doux']['rho']:.2f} ({a['V']['doux']['verdict']})")
    else:
        print(f"[SYNTHÈSE] {lab} : {a['these']}")
print(f"[HYPOTHÈSE] le jeton choqué produit une valeur ‖W_O v‖ plus grande : {hyp} "
      f"(SAILLANCE pour {n_s} modèles, ÉQUIVALENT ou SUPPRESSION pour {n_r}, sur {len(vs)} ; prédiction : SOUTENUE 55 %)")
ALL_OUT = {"hypothese_valeur": hyp}
ALL["_synthese"] = ALL_OUT
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

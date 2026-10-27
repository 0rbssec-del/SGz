# %% [markdown]
# # SGZ A1j — Format conversation : quel facteur amortit le choc ? (consigne, gabarit, tour de l'assistant)
#
# **But.** A1g : au format conversation, l'effet direct du choc fort sur l'entropie de Qwen3-4B tombe de +0,051 (texte brut,
# A1f) à +0,006 (÷ 9) à T = 0,6. Le format changeait trois choses à la fois. A1j les sépare et mesure aussi l'entropie de
# base du témoin, pour distinguer un vrai amortissement d'un simple effet d'échelle (modèle plus confiant).
#
# **Cinq formats, même modèle (Qwen/Qwen3-4B), mêmes 768 prompts, mêmes jetons de choc (rang 1000–5000), T = 0,6.**
#
# | Format | Gabarit de conversation | Consigne « Continue ce texte : » | Où le modèle écrit |
# |---|---|---|---|
# | brut (= A1f) | non | non | suite directe du texte |
# | consigne | non | oui (texte brut) | suite directe du texte |
# | prerempli | oui | oui | suite directe du texte, **dans** le tour de l'assistant (prompt prérempli) |
# | complet (= A1g) | oui | oui | nouveau tour de l'assistant (le texte est dans le message de l'utilisateur) |
# | gabarit | oui | non | nouveau tour de l'assistant (message de l'utilisateur = texte seul) |
#
# La chaîne brut → consigne → prerempli → complet passe de A1f à A1g en changeant **un seul** facteur à chaque pas :
# **C** (consigne) = consigne / brut ; **G** (gabarit) = prerempli / consigne ; **R** (nouveau tour) = complet / prerempli.
# Le produit des trois rapports vaut complet / brut. Le format « gabarit » donne une seconde mesure de C (complet / gabarit),
# sous gabarit.
#
# ## Critères (fixés avant toute exécution)
#
# Mesures par format (passes rejouées, comme A1c–A1g) : effet direct **D** = moyenne sur k = 2–31 de S(choc fort) − S(témoin) ;
# entropie de base **S0** = moyenne sur k = 2–31 de S(témoin) ; effet relatif **Dr** = D / S0.
# - **Rapport d'un facteur** r = D(format X) / D(format Y), bootstrap apparié sur les prompts (B = 1000, mêmes tirages
#   pour tous les formats). SANS EFFET si IC95 ⊂ [0,8 ; 1,2] ; RÉDUIT si IC95 < 1 ; AUGMENTÉ si IC95 > 1 ; sinon
#   INDÉTERMINÉ. NON TESTABLE si l'IC95 de D(Y) contient 0. Même règle pour Dr.
# - **Effet d'échelle** pour un facteur : D RÉDUIT et Dr SANS EFFET → ÉCHELLE (le choc pèse autant en relatif) ;
#   D RÉDUIT et Dr RÉDUIT → AMORTISSEMENT ; sinon la lecture suit les deux verdicts.
# - **Part de chaque facteur** dans l'écart total (en log) : part_i = log r_i / log r_total (C, G, R ; somme = 1).
#   DOMINANT si la borne basse de l'IC95 de part_i > 0,5 ; sinon aucun facteur dominant (PARTAGÉ).
# - **Ancrages** : format brut, D comparé à A1f (+0,0510) ; format complet, trajectoires identiques à A1g (MD5) et D
#   comparé à A1g (+0,0057). Un écart n'invalide pas les contrastes (tous mesurés ici) mais est signalé.
# - Mesures secondaires, sans verdict de thèse : F (trajectoire), A et N (lecture du jeton choqué), E (correction explicite),
#   par format, avec les règles de A1g.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# R (nouveau tour) DOMINANT 40 % ; G (gabarit) DOMINANT 20 % ; C (consigne) DOMINANT 10 % ; PARTAGÉ 30 %.
# Écart total complet / brut : AMORTISSEMENT (Dr réduit aussi) 75 %, ÉCHELLE seule 15 %.
#
# ## Limites connues
#
# - Un seul modèle instruct, une seule température ; la consigne et le gabarit sont ceux de A1g (une seule formulation).
# - « prerempli » met le texte de l'utilisateur dans la bouche de l'assistant : situation rare à l'entraînement.
# - Les jetons de choc sont choisis par rang dans chaque format, donc ce ne sont pas les mêmes jetons d'un format à l'autre.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : ~20 min par format, ~1 h 45 en tout [estimation].
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[ANCRAGE]`, `[VÉRIF]`, `[BASE]`, `[FACTEUR]`, `[PART]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1j" if os.path.isdir("/kaggle/working") else "sgz_a1j_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]                                    # une seule température (A1g : 0,6 et 0,8 concordants)
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 768, 160, 64
RANK_LO, RANK_HI = 1000, 5000
W_MAIN, MID_D = (8, 31), (2, 31)
MID_F, MARGE_REL = (2, 31), 0.3
WINS = [(2, 7), (8, 31), (32, 95)]
MARGE = 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                   # même graine que A1c : mêmes trajectoires
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))    # par format
ARMS_R = ["temoin", "fort", "doux"]
A1G_MD5 = {"temoin": "f64daaa355e36bb1d69607045a487aca", "fort": "e905ff10af38fe69072ba351b67d4c52"}   # A1g, T = 0,6
REF_D = {"brut": (0.0510, "A1f"), "complet": (0.0057, "A1g")}          # effet direct du choc fort à T = 0,6
FORMATS = ["brut", "consigne", "prerempli", "complet", "gabarit"]
FACTEURS = [("C", "consigne", "brut"), ("G", "prerempli", "consigne"), ("R", "complet", "prerempli"),
            ("C sous gabarit", "complet", "gabarit"), ("total", "complet", "brut")]
NOMS_F = {"C": "consigne", "G": "gabarit de conversation", "R": "nouveau tour de l'assistant",
          "C sous gabarit": "consigne, sous gabarit", "total": "format conversation complet"}
MODELES_A_LANCER = ["Qwen/Qwen3-4B"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
REF_4B = {"I/D": {0.6: 0.55, 0.8: 0.47}, "rho_A": {0.6: 1.13, 0.8: 1.04}, "rho_N": {0.6: 2.11, 0.8: 1.92},
          "D": {0.6: 0.0510, 0.8: 0.0494}}                    # Qwen3-4B instruct en texte brut (A1f)
CORR_RE = {"fr": re.compile(r"\b(pardon|d[ée]sol[ée]e?|je (?:veux|voulais) dire|oups|excusez|correction|faute de frappe)\b", re.I),
           "en": re.compile(r"\b(sorry|i mean|i meant|oops|typo|correction|apolog\w*|let me rephrase)\b", re.I)}
CONSIGNE = {"fr": "Continue ce texte :\n\n{p}", "en": "Continue this text:\n\n{p}"}
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


def facteur_verdict(x, y, boots, interpretable=True):
    """Rapport r = moyenne(x) / moyenne(y) entre deux formats (mêmes prompts), bootstrap apparié.
    x, y : tableaux (D, S0) par prompt ; renvoie le verdict sur D et sur Dr = D / S0."""
    dX, sX = x
    dY, sY = y
    out = {}
    cY = ci([dY[ix].mean() for ix in boots])
    for nom, f in (("D", lambda ix: dX[ix].mean() / dY[ix].mean()),
                   ("Dr", lambda ix: (dX[ix].mean() / sX[ix].mean()) / (dY[ix].mean() / sY[ix].mean()))):
        r = float(f(np.arange(len(dX))))
        c = ci([f(ix) for ix in boots])
        if not interpretable:
            v = "NON INTERPRÉTABLE"
        elif cY[0] <= 0 <= cY[1]:
            v = "NON TESTABLE (effet direct absent dans le format de référence)"
        elif c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
            v = "SANS EFFET"
        elif c[1] < 1:
            v = "RÉDUIT"
        elif c[0] > 1:
            v = "AUGMENTÉ"
        else:
            v = "INDÉTERMINÉ"
        out[nom] = {"r": r, "ci": c, "verdict": v}
    vD, vR = out["D"]["verdict"], out["Dr"]["verdict"]
    out["lecture"] = ("ÉCHELLE" if vD == "RÉDUIT" and vR == "SANS EFFET" else
                      "AMORTISSEMENT" if vD == "RÉDUIT" and vR == "RÉDUIT" else f"D {vD} ; Dr {vR}")
    return out


def parts_chaine(dd, boots, interpretable=True):
    """Part de C, G et R dans log(complet / brut), par bootstrap ; DOMINANT si borne basse > 0,5."""
    def logs(ix):
        m = {f: dd[f][0][ix].mean() for f in ("brut", "consigne", "prerempli", "complet")}
        with np.errstate(divide="ignore", invalid="ignore"):
            lc, lg, lr = np.log(m["consigne"] / m["brut"]), np.log(m["prerempli"] / m["consigne"]), np.log(m["complet"] / m["prerempli"])
        return lc, lg, lr, lc + lg + lr
    full = logs(np.arange(len(dd["brut"][0])))
    bs = np.array([logs(ix) for ix in boots], dtype=float)
    ct = ci(bs[:, 3])
    out = {"log_total": float(full[3]), "ci_log_total": ct}
    for i, f in enumerate(("C", "G", "R")):
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = bs[:, i] / bs[:, 3]
        c = ci(sh)
        if not interpretable:
            v = "NON INTERPRÉTABLE"
        elif ct[0] <= 0 <= ct[1]:
            v = "NON TESTABLE (écart total non significatif)"
        elif c[0] > 0.5:
            v = "DOMINANT"
        else:
            v = "NON DOMINANT"
        out[f] = {"part": float(full[i] / full[3]) if full[3] != 0 else float("nan"), "ci": c, "verdict": v,
                  "valides": float(np.isfinite(sh).mean())}
    doms = [f for f in ("C", "G", "R") if out[f]["verdict"] == "DOMINANT"]
    out["facteur_principal"] = doms[0] if len(doms) == 1 else ("PARTAGÉ" if all(out[f]["verdict"] == "NON DOMINANT"
                                                                                 for f in ("C", "G", "R")) else "INDÉTERMINÉ")
    if not interpretable:
        out["facteur_principal"] = "NON INTERPRÉTABLE"
    return out


def synth_formats(rC, rG, rR, Pn=Pp, d0=0.05, s0=0.5, sr=(1, 1, 1, 1)):
    """Formats synthétiques : D par prompt = d0 × produit des rapports × bruit ; S0 réduit par sr."""
    base = rng_pc.lognormal(np.log(d0), 0.4, Pn)
    fac = {"brut": 1.0, "consigne": rC, "prerempli": rC * rG, "complet": rC * rG * rR}
    return {f: (fac[f] * base * rng_pc.lognormal(0, 0.3, Pn) + rng_pc.normal(0, 0.01, Pn),
                s0 * sr[i] * rng_pc.lognormal(0, 0.1, Pn)) for i, f in enumerate(fac)}


ff = synth_formats(1.0, 1.0, 0.11)
ff_e = synth_formats(1.0, 1.0, 0.11, sr=(1, 1, 1, 0.11))
for name, got, exp in (
        ("facteur réduit", facteur_verdict(ff["complet"], ff["prerempli"], bt)["D"]["verdict"], "RÉDUIT"),
        ("facteur sans effet", facteur_verdict(ff["consigne"], ff["brut"], bt)["D"]["verdict"], "SANS EFFET"),
        ("facteur augmenté", facteur_verdict(ff["prerempli"], ff["complet"], bt)["D"]["verdict"], "AUGMENTÉ"),
        ("lecture échelle", facteur_verdict(ff_e["complet"], ff_e["prerempli"], bt)["lecture"], "ÉCHELLE"),
        ("lecture amortissement", facteur_verdict(ff["complet"], ff["prerempli"], bt)["lecture"], "AMORTISSEMENT"),
        ("part R dominante", parts_chaine(ff, bt)["facteur_principal"], "R"),
        ("parts partagées", parts_chaine(synth_formats(0.48, 0.48, 0.48), bt)["facteur_principal"], "PARTAGÉ")):
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
                    for j, r in enumerate(rows_p):
                        rk = int(min(forced_rank[r], order.shape[1] - 2))
                        if int(order[j, rk]) == int(nxt[j]):     # jamais le jeton du témoin
                            rk += 1
                        nxt[n + j] = order[j, rk]
                        rang[r] = rk
                        soft[r] = int(order[j, 0]) if int(order[j, 0]) != int(nxt[j]) else int(order[j, 1])
                return nxt

            S, tk = g.sample([enc[r] for r in rows_p] * 2, rows_p * 2, T, GEN_LEN, 0, force)
            for ai, a in enumerate(("temoin", "fort")):
                sl = slice(ai * n, (ai + 1) * n)
                res[("S", T, a)][rows_p], res[("tok", T, a)][rows_p] = S[sl], tk[sl]
            g.progress(time.time() - tl, n_lots)
        res[("soft", T)] = soft
        ref, fo = res[("tok", T, "temoin")], res[("tok", T, "fort")]
        same = np.array_equal(ref[:, :T0], fo[:, :T0])
        chg = float(np.mean(ref[:, T0] != fo[:, T0]))
        dsoft = float(np.mean(soft != ref[:, T0]))
        print(f"[VÉRIF] {g.label} T={T} : identique avant t0 : {same} ; jeton changé à t0 : fort {chg:.0%}, doux {dsoft:.0%} ; "
              f"rang médian du choc fort {int(np.median(rang))}", flush=True)
        if not same or chg < 0.99 or dsoft < 0.99:
            raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
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
        self.v, self.A, self.N, self.Ah = ([None] * self.nL for _ in range(4))
        self.handles = []
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(partial(self.pre, l), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(partial(self.post, l)))

    def pre(self, l, mod, args, kwargs):
        if not self.active:
            return
        hs = kwargs.get("hidden_states", args[0] if args else None)
        b = torch.arange(hs.shape[0], device=hs.device)
        self.v[l] = mod.v_proj(hs[b, self.kidx.to(hs.device)]).float()        # valeurs du jeton t₀ (après la norme)

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
        flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
    A = torch.stack(probe.A, 1).cpu().numpy()                                  # (n, couches, K)
    N = torch.stack(probe.N, 1).cpu().numpy()
    Ah = torch.stack(probe.Ah, 1).cpu().numpy()                                # (n, couches, têtes)
    return flat.reshape(n, Kp).astype(np.float32), A, N, Ah


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
        for b0 in range(0, P, nb):
            rows = list(range(b0, min(P, b0 + nb)))
            ct = [ref[r, :GEN_LEN - 1].tolist() for r in rows]
            cf = [c[:T0] + [int(fo[r, T0])] + c[T0 + 1:] for c, r in zip(ct, rows)]
            cd = [c[:T0] + [int(soft[r])] + c[T0 + 1:] for c, r in zip(ct, rows)]
            probe.check = b0 == 0
            S, A, N, Ah = replay_attn(g, probe, [enc[r] for r in rows] * 3, ct + cf + cd)
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
# ---------- Boucle principale : un modèle, cinq formats ----------
interpretable = not SMOKE_TEST
ALL = {"formats": {}}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1j.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def chat(tok, user):
    """Gabarit de conversation officiel, réflexion désactivée ; gabarit factice pour le tokenizer jouet (SMOKE_TEST)."""
    if SMOKE_TEST or not hasattr(tok, "apply_chat_template"):
        return f"<U>{user}<A>"
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode_format(tok, fmt):
    enc = []
    for p, l in zip(PROMPTS, LANG):
        c = CONSIGNE[l].format(p=p)
        if fmt == "brut":
            e = tok.encode(p)                                          # comme A1f
        elif fmt == "consigne":
            e = tok.encode(c)
        elif fmt == "prerempli":
            e = tok.encode(chat(tok, c) + p, add_special_tokens=False)
        elif fmt == "complet":
            e = tok.encode(chat(tok, c), add_special_tokens=False)     # comme A1g
        elif fmt == "gabarit":
            e = tok.encode(chat(tok, p), add_special_tokens=False)
        else:
            raise ValueError(fmt)
        enc.append(e[-16:] if SMOKE_TEST else e)
    return enc


label, ids = MODELS[0]
t_all = time.time()
model = tok = None
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
    raise RuntimeError("modèle non chargé : rien à mesurer")
if getattr(tok, "pad_token_id", None) is None:
    tok.pad_token = tok.eos_token
ATTN_GEN = getattr(model.config, "_attn_implementation", None) or "sdpa"
print(f"[CONFIG] attention pendant la génération : {ATTN_GEN} ; passes rejouées : eager")
PAR = {}                                                               # (D, S0) par prompt, pour les contrastes
T = T_MAIN[0]
for fmt in FORMATS:
    t_start = time.time()
    g = probe = res = None
    try:
        enc = encode_format(tok, fmt)
        print(f"\n[FORMAT] {fmt} : exemple {tok.decode(enc[0])!r} ; longueur médiane du contexte "
              f"{int(np.median([len(e) for e in enc]))} jetons", flush=True)
        set_attention(model, ATTN_GEN)
        g = Gen(model, tok, f"{label}|{fmt}")
        try:
            res = run_main(g, enc)
        except FloatingPointError as e:
            print(f"[VÉRIF] {fmt} : {e} en fp16 → rechargement en fp32 pour ce format et les suivants", flush=True)
            del g
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model, tok, mode = load_model(ids[0], fp32=True)
            if getattr(tok, "pad_token_id", None) is None:
                tok.pad_token = tok.eos_token
            set_attention(model, ATTN_GEN)
            print(f"[CHARGEMENT] {label} : {ids[0]} en {mode}", flush=True)
            g = Gen(model, tok, f"{label}|{fmt}")
            res = run_main(g, enc)
        if fmt == "complet" and not SMOKE_TEST:
            ok_t = md5(res[("tok", T, "temoin")]) == A1G_MD5["temoin"]
            ok_f = md5(res[("tok", T, "fort")]) == A1G_MD5["fort"]
            print(f"[ANCRAGE] complet : trajectoires identiques à A1g (MD5) : témoin {ok_t}, choc fort {ok_f}"
                  + ("" if ok_t and ok_f else " (écart numérique possible ; les contrastes restent valides)"), flush=True)
        set_attention(model, "eager")
        probe = AttnProbe(model)
        res = run_replay(g, probe, enc, res)
        for hnd in probe.handles:
            hnd.remove()
        a = analyse(res, interpretable)
        R = (res[("Srej", T, "fort")] - res[("Srej", T, "temoin")])[:, T0 + 1:].astype(float)
        S0 = win(res[("Srej", T, "temoin")][:, T0 + 1:].astype(float), MID_D)
        PAR[fmt] = (win(R, MID_D), S0)
        Rd = (res[("Srej", T, "doux")] - res[("Srej", T, "temoin")])[:, T0 + 1:].astype(float)
        hit = {arm: np.array([bool(CORR_RE[LANG[r]].search(tok.decode(res[("tok", T, arm)][r, T0 + 1:].tolist())))
                              for r in range(len(enc))], float) for arm in ("temoin", "fort")}
        a["base"] = {"S0": float(S0.mean()), "S_avant_t0": float(res[("Srej", T, "temoin")][:, T0 - 32:T0].mean()),
                     "Dr": float(PAR[fmt][0].mean() / S0.mean()), "D_doux": float(win(Rd, MID_D).mean()),
                     "correction_fort": float(hit["fort"].mean()), "correction_temoin": float(hit["temoin"].mean())}
        a["minutes"] = (time.time() - t_start) / 60
        ALL["formats"][fmt] = a
        f_, d_, b_ = a["F"][T], a["D"][T]["fort"], a["base"]
        ref = REF_D.get(fmt)
        print(f"[BASE] {fmt:<9} : S0 (témoin, k={MID_D[0]}–{MID_D[1]}) = {b_['S0']:.3f} ; D fort = {d_['D']:+.4f} "
              f"IC95 [{d_['ci'][0]:+.4f} ; {d_['ci'][1]:+.4f}] ; Dr = D/S0 = {b_['Dr']:+.4f} ; D doux = {b_['D_doux']:+.4f}"
              + (f" ; référence {ref[1]} : {ref[0]:+.4f}" if ref and not SMOKE_TEST else ""), flush=True)
        if ref and not SMOKE_TEST:
            dans = d_["ci"][0] <= ref[0] <= d_["ci"][1]
            print(f"[ANCRAGE] {fmt} : D de {ref[1]} ({ref[0]:+.4f}) dans l'IC95 : {dans}"
                  + ("" if dans else " (écart signalé ; les contrastes, mesurés ici, restent valides)"), flush=True)
        print(f"[F] {fmt:<9} : {f_['verdict']} ; I = {f_['I']:+.4f} IC95 [{f_['ci_I'][0]:+.4f} ; {f_['ci_I'][1]:+.4f}] ; "
              f"I/D = {f_['rapport']:+.2f}")
        for meas in ("A", "N"):
            r = a[meas][T]["fort"]
            print(f"[{meas}] {fmt:<9} : choc fort ρ = {r['rho']:.3f} IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}] ({r['verdict']}) ; "
                  f"têtes ρ_A < 0,8 : {a['tetes'][f'{T}_part']['fort<0.8']:.1%}" if meas == "A" else
                  f"[{meas}] {fmt:<9} : choc fort ρ = {r['rho']:.3f} IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}] ({r['verdict']})")
        print(f"[COURBE] {fmt:<9} : D fort par tranches de k (1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–95) : "
              + " ".join(f"{v:+.3f}" for v in d_["courbe"]))
        print(f"[E] {fmt:<9} : formules de correction, choc {b_['correction_fort']:.1%}, témoin {b_['correction_temoin']:.1%}")
        print(f"[EXTRAIT] {fmt} prompt 0 : {tok.decode(res[('tok', T, 'fort')][0, T0 - 8:T0 + 40].tolist())!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_{fmt}_a1j.npz"),
                            **{f"{k}_{arm}": res[(k, T, arm)] for k in ("S", "tok") for arm in ("temoin", "fort")},
                            soft=res[("soft", T)],
                            **{f"{k}_{arm}": res[(k, T, arm)].astype(np.float16 if k == "Ah" else np.float32)
                               for k in ("Srej", "A", "N", "Ah") for arm in ARMS_R})
        print(f"[DURÉE] {fmt} : {a['minutes']:.1f} min", flush=True)
    except Exception as e:
        print(f"[ERREUR] {fmt} : {type(e).__name__}: {str(e)[:500]}", flush=True)
        ALL["formats"][fmt] = {"erreur": f"{type(e).__name__}: {str(e)[:300]}"}
    del g, probe, res
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ecrire_resume()

# %%
# ---------- Contrastes entre formats ----------
ok = interpretable and POSITIVE_CONTROL_OK
P = len(PROMPTS)
rngF = np.random.default_rng(SEED + 11)
bootsF = [rngF.integers(0, P, P) for _ in range(BOOT_N)]
print(f"\n{'=' * 100}\n[RÉSULTATS] {label}, T = {T}, contrastes entre formats (même prompts, bootstrap apparié)")
ALL["facteurs"] = {}
for nom, x, y in FACTEURS:
    if x not in PAR or y not in PAR:
        ALL["facteurs"][nom] = {"lecture": "NON TESTABLE (format manquant)"}
        print(f"[FACTEUR] {nom:<15} ({x} / {y}) : NON TESTABLE (format manquant)")
        continue
    v = facteur_verdict(PAR[x], PAR[y], bootsF, ok)
    ALL["facteurs"][nom] = v
    print(f"[FACTEUR] {nom:<15} ({NOMS_F[nom]} ; {x} / {y}) : D ×{v['D']['r']:.2f} IC95 [{v['D']['ci'][0]:.2f} ; "
          f"{v['D']['ci'][1]:.2f}] {v['D']['verdict']} ; Dr ×{v['Dr']['r']:.2f} IC95 [{v['Dr']['ci'][0]:.2f} ; "
          f"{v['Dr']['ci'][1]:.2f}] {v['Dr']['verdict']} → {v['lecture']}")
if all(f in PAR for f in ("brut", "consigne", "prerempli", "complet")):
    pc = parts_chaine(PAR, bootsF, ok)
    ALL["parts"] = pc
    print(f"[PART] écart total log(complet / brut) = {pc['log_total']:+.2f} IC95 [{pc['ci_log_total'][0]:+.2f} ; "
          f"{pc['ci_log_total'][1]:+.2f}]")
    for f in ("C", "G", "R"):
        r = pc[f]
        print(f"[PART] {f} ({NOMS_F[f]}) : {r['part']:+.0%} de l'écart IC95 [{r['ci'][0]:+.0%} ; {r['ci'][1]:+.0%}] : {r['verdict']}")
    print(f"[SYNTHÈSE] facteur principal de l'amortissement : {pc['facteur_principal']} "
          f"(prédiction : R 40 %, G 20 %, C 10 %, PARTAGÉ 30 %)")
else:
    ALL["parts"] = {"facteur_principal": "NON TESTABLE (format manquant)"}
    print("[SYNTHÈSE] facteur principal : NON TESTABLE (format manquant)")
if "total" in ALL["facteurs"] and "D" in ALL["facteurs"]["total"]:
    print(f"[SYNTHÈSE] écart total complet / brut : {ALL['facteurs']['total']['lecture']} "
          f"(prédiction : AMORTISSEMENT 75 %, ÉCHELLE 15 %)")
if "C" in ALL["facteurs"] and "C sous gabarit" in ALL["facteurs"] and "D" in ALL["facteurs"]["C"] \
        and "D" in ALL["facteurs"]["C sous gabarit"]:
    print(f"[SYNTHÈSE] consigne sans gabarit ×{ALL['facteurs']['C']['D']['r']:.2f}, sous gabarit "
          f"×{ALL['facteurs']['C sous gabarit']['D']['r']:.2f} (descriptif : interaction consigne × gabarit)")
ALL["minutes"] = (time.time() - t_all) / 60
print(f"[DURÉE] totale : {ALL['minutes']:.1f} min")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

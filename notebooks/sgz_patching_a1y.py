# %% [markdown]
# # SGZ A1y — Tracé causal du jeton choqué, couche par couche : où l'amplification est-elle lue ?
#
# **Point de départ.** La trajectoire amplifie un choc (I > 0, 24 verdicts sur 24 de A1c à A1m, reproduit dans A1u).
# Cinq explications simples sont tombées : excès de valeur, redistribution entre têtes, routage vers les couches profondes,
# divergence dans le flux résiduel, direction unique (en reprise, A1u2). Une suggestion de Kimi (29 oct.) n'avait jamais
# été faite : le patching d'activations dans les deux sens. Son code mesurait une préférence entre deux jetons à un seul
# pas, sur une seule paire de phrases ; il ne pouvait pas voir I, un effet de trajectoire. A1y fait le patching sur la
# génération elle-même, avec la mesure I et 768 prompts.
#
# **Idée.** Avant t₀, témoin et choc sont identiques ; ils ne diffèrent que par le jeton t₀. Remplacer l'état caché de la
# position t₀ à la sortie de la couche l :
# - **débruitage** : dans le choc, par celui du témoin → le choc n'est visible qu'aux couches 0 … l (les positions
#   suivantes le lisent par l'attention de ces couches seulement) ; l = −1 (plongements) : tout est restauré ;
# - **bruitage** : dans le témoin, par celui du choc → le choc n'est visible qu'aux couches l + 1 … L − 1 et dans la loi
#   du jeton t₀ + 1 ; l = −1 : le choc entier.
#
# **Dispositif.** Qwen3-1.7B-Base (celui d'A1u), 768 prompts, T = 0,6, 96 jetons, choc de rang 1000–5000 à t₀ = 64, même
# graine, même tirage des rangs et même bruit que A1c–A1u. Pour chaque direction et chaque couche de la grille
# l ∈ {−1, 1, 3, …, 27} : la suite libre est régénérée après t₀ avec le crochet actif (même bruit que le témoin), puis
# passes rejouées de la suite libre et du rejoué (suite du témoin), crochet actif. Le témoin sans crochet sert de référence.
#
# ## Mesures (par prompt, moyenne sur k = 2–31)
#
# D = S(rejoué) − S(témoin) ; I = S(libre) − S(rejoué) (règle F de A1c), sans crochet (I₀) et sous chaque crochet.
# Part de l'amplification qui subsiste : f_déb(l) = I_déb(l) / I₀ et f_bru(l) = I_bru(l) / I₀.
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# - **Préalable** : I₀ doit être AMPLIFICATION (règle F) ; sinon tout est NON TESTABLE.
# - **Contrôles du crochet** (bloquants) : débruitage à l = −1 → la suite libre redonne le témoin pour ≥ 90 % des prompts et
#   |D| < 0,002 ; débruitage à la dernière couche → la loi du jeton t₀ + 1 redevient celle du témoin (écart moyen < 0,005 ;
#   le témoin rejoué contre généré diffère déjà de ~1,5e-3 en moyenne et jusqu'à 3e-2 en fp16) ; bruitage à l = −1 → la suite libre redonne le choc sans crochet pour ≥ 90 % des prompts.
# - **Y1 (principal), profondeur de lecture** : l₅₀ = plus petite couche de la grille telle que f_déb(l) ≥ 0,5 (la moitié de
#   l'amplification revient dès que le choc est visible jusqu'à la couche l). Sur 28 couches : PRÉCOCE si la borne haute de
#   l'IC95 de l₅₀ < 28/3 ; TARDIVE si la borne basse ≥ 2 × 28/3 ; INTERMÉDIAIRE si l'IC95 est entre les deux ; NON ATTEINT
#   si la borne haute de f_déb(27) < 0,5 (l'amplification passe alors surtout par la loi du jeton t₀ + 1, la seule chose
#   que le débruitage à la dernière couche corrige) ; sinon INDÉTERMINÉE.
# - **Y2, additivité** : A = moyenne sur les couches l ≥ 0 de la grille de f_déb(l) + f_bru(l). Lecture dans une seule
#   bande de couches → A ≈ 1 ; information disponible à plusieurs profondeurs → A > 1 ; il faut à la fois les couches
#   basses et hautes → A < 1. ADDITIF si IC95 ⊂ [0,8 ; 1,2] ; REDONDANT si la borne basse > 1,2 ; CONJONCTIF si la borne
#   haute < 0,8 ; sinon INDÉTERMINÉ.
# - **Synthèse, thèse « l'amplification est lue à partir du jeton choqué dans une bande de couches précise »** : SOUTENUE
#   si Y2 = ADDITIF et Y1 ∈ {PRÉCOCE, INTERMÉDIAIRE, TARDIVE} ; RÉFUTÉE si Y2 ∈ {REDONDANT, CONJONCTIF} ; sinon
#   INDÉTERMINÉE.
# - **Descriptifs** : courbes f_déb, f_bru, D_déb / D₀ et D_bru / D₀ avec IC95 ; part des suites libres identiques au
#   témoin ou au choc ; lentille des logits (logit lens : entropie de Tsallis de head(norm(h_l)), rejoué − témoin, aux
#   couches paires et aux k ∈ {1, 2, 4, 8, 16}) et rapport des normes ‖h_l(rejoué)‖ / ‖h_l(témoin)‖ par couche, sur 384
#   prompts (idées de Kimi, en descriptif).
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# I₀ AMPLIFICATION 90 %. Y1 : INTERMÉDIAIRE 35 %, PRÉCOCE 30 %, TARDIVE 10 %, NON ATTEINT 10 %, INDÉTERMINÉE 15 %.
# Y2 : ADDITIF 35 %, REDONDANT 30 %, CONJONCTIF 10 %, INDÉTERMINÉ 25 %. Synthèse SOUTENUE 30 %, RÉFUTÉE 40 %.
#
# ## Limites connues
#
# - Puissance : I₀ est petit (IC95 d'A1u : ±45 % de sa valeur) ; toutes les parts f sont divisées par le même I₀, donc
#   la forme des courbes est mieux connue que leur niveau ; l₅₀ peut rester INDÉTERMINÉE.
# - Le patching porte sur la seule position t₀ ; un mécanisme qui passe par les positions suivantes dès les premières
#   couches n'est vu qu'à travers elles.
# - Une couche sur deux ; un modèle ; anomalies artificielles.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~55 min [estimation].
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[F]`, `[COURBE]`, `[Y1]`, `[Y2]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1y" if os.path.isdir("/kaggle/working") else "sgz_a1y_out")
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
KS_LENS = [1, 2, 4, 8, 16]
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1u2
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1U = {"Qwen3-1.7B-Base": {"I": 0.03651, "D": 0.07735}}                       # sans ablation, 768 prompts
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1 = int(os.environ.get("SMOKE_N", "16")), 48, 12
    RANK_LO, RANK_HI = 20, 80
    MID = (2, 12)
    KS_LENS = [1, 2, 4, 8]
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
NSUITE = NPOS - 2                                # jetons libres après t₀ : t₀ + 1 … t₀ + 30
LONG = T1 + 1 + NSUITE                           # textes rejoués : jetons 0 … t₀ + 30
assert LONG <= GEN_LEN
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T1} fenêtre k={MID[0]}–{MID[1]} ; patching de la "
      f"position t₀, débruitage et bruitage ; modèles={[m[0] for m in MODELS]}")
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


def grille_de(nl):
    """Couches patchées : −1 (plongements) puis une couche sur deux, la dernière comprise."""
    return [-1] + list(range(1, nl, 2)) + ([nl - 1] if (nl - 1) % 2 == 0 else [])


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


def courbe_f(Ic, I0, grille, ix=None):
    """Parts f(l) = Ī(l) / Ī₀ ; ix : rééchantillon des prompts."""
    sel = (lambda x: x) if ix is None else (lambda x: x[ix])
    d0 = sel(I0).mean()
    return np.array([sel(Ic[l]).mean() / d0 for l in grille])


def l50_de(f, grille, nl):
    for l, x in zip(grille, f):
        if x >= 0.5:
            return l
    return nl                                    # jamais atteint


def regle_y1(Id, I0, grille, nl, boots, amp, interpretable=True):
    f = courbe_f(Id, I0, grille)
    lb = [l50_de(courbe_f(Id, I0, grille, ix), grille, nl) for ix in boots]
    c = (float(np.percentile(lb, 2.5)), float(np.percentile(lb, 97.5)))
    fl = ci([Id[grille[-1]][ix].mean() / I0[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif not amp:
        v = "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"
    elif fl[1] < 0.5:
        v = "NON ATTEINT (par la loi du jeton t₀ + 1)"
    elif c[1] < nl / 3:
        v = "PRÉCOCE"
    elif c[0] >= 2 * nl / 3:
        v = "TARDIVE"
    elif c[0] >= nl / 3 and c[1] < 2 * nl / 3:
        v = "INTERMÉDIAIRE"
    else:
        v = "INDÉTERMINÉE"
    return {"f": [float(x) for x in f], "l50": l50_de(f, grille, nl), "ci": c, "f_derniere_ci": fl, "verdict": v}


def regle_y2(Id, In, I0, couches, boots, amp, interpretable=True):
    A = lambda ix=None: float(np.mean(courbe_f(Id, I0, couches, ix) + courbe_f(In, I0, couches, ix)))
    c = ci([A(ix) for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif not amp:
        v = "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"
    elif c[0] >= 0.8 and c[1] <= 1.2:
        v = "ADDITIF"
    elif c[0] > 1.2:
        v = "REDONDANT"
    elif c[1] < 0.8:
        v = "CONJONCTIF"
    else:
        v = "INDÉTERMINÉ"
    return {"A": A(), "ci": c, "verdict": v}


def synthese(y1, y2, interpretable=True):
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if y2 == "ADDITIF" and y1 in ("PRÉCOCE", "INTERMÉDIAIRE", "TARDIVE"):
        return "SOUTENUE"
    if y2 in ("REDONDANT", "CONJONCTIF"):
        return "RÉFUTÉE"
    return "INDÉTERMINÉE"


# Témoins positifs : courbes synthétiques de taille connue (28 couches, 768 prompts)
rng_pc = np.random.default_rng(SEED + 1)
Pp, nl_pc = 768, 28
gr_pc = grille_de(nl_pc)
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
I0_pc = 0.04 + rng_pc.normal(0, 0.1, Pp)
bruit = lambda: (lambda x: x - x.mean())(rng_pc.normal(0, 0.01, Pp))
marche = lambda l0: {l: (1.0 if l >= l0 else 0.0) * I0_pc + bruit() for l in gr_pc}
comp = lambda Id: {l: I0_pc - Id[l] + bruit() for l in gr_pc}
const = lambda a: {l: (a if l >= 0 else 0.0) * I0_pc + bruit() for l in gr_pc}
pos = [l for l in gr_pc if l >= 0]
y1 = lambda Id: regle_y1(Id, I0_pc, gr_pc, nl_pc, bt, True)["verdict"]
y2 = lambda Id, In: regle_y2(Id, In, I0_pc, pos, bt, True)["verdict"]
checks = [("Y1 précoce", y1(marche(5)), "PRÉCOCE"),
          ("Y1 intermédiaire", y1(marche(13)), "INTERMÉDIAIRE"),
          ("Y1 tardive", y1(marche(21)), "TARDIVE"),
          ("Y1 non atteint", y1(const(0.2)), "NON ATTEINT (par la loi du jeton t₀ + 1)"),
          ("Y2 additif", y2(marche(13), comp(marche(13))), "ADDITIF"),
          ("Y2 redondant", y2(const(1.0), const(1.0)), "REDONDANT"),
          ("Y2 conjonctif", y2(const(0.0), const(0.0)), "CONJONCTIF"),
          ("sans amplification", regle_y1(marche(5), I0_pc, gr_pc, nl_pc, bt, False)["verdict"],
           "NON TESTABLE (I₀ n'est pas AMPLIFICATION)"),
          ("synthèse soutenue", synthese("INTERMÉDIAIRE", "ADDITIF"), "SOUTENUE"),
          ("synthèse réfutée", synthese("PRÉCOCE", "REDONDANT"), "RÉFUTÉE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<20} : {got:<42} (attendu {att})")
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
# ---------- Génération (identique à A1c–A1u ; crochet de patching au préremplissage) ----------
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
    def sample(self, ctxs, prompt_of, T, nsteps, step0, force=None):
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
# ---------- Choc à t₀ (comme A1u), crochet de patching, génération et passes rejouées ----------
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


class Patcheur:
    """Remplace l'état caché d'une position p à la sortie de la couche l (l = −1 : sortie des plongements) par val (n, d).
    Pendant la génération, la valeur val_gen ne s'applique qu'à la passe de préremplissage (dernière position = t₀)."""

    def __init__(self, model):
        self.l, self.p, self.val, self.val_gen = None, None, None, None
        self.handles = [model.model.embed_tokens.register_forward_hook(self._hook(-1))]
        self.handles += [c.register_forward_hook(self._hook(i)) for i, c in enumerate(model.model.layers)]

    def _hook(self, i):
        def f(mod, args, out):
            if self.val is None or self.p is None or i != self.l:
                return None
            h = sortie(out).clone()
            h[:, self.p] = self.val.to(device=h.device, dtype=h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def generer(g, enc, idx, tokm, x, val=None):
    """Suite libre après prompt + tokm[:, :t₀] + x (jeton t₀) : jetons t₀ + 1 … t₀ + NSUITE, bruit du témoin ; avec val
    (P, d), l'état caché de t₀ est remplacé à la couche g.patch.l."""
    P = len(idx)
    tk = np.zeros((P, NSUITE), np.int64)
    for b0 in range(0, P, g.budget):
        loc = list(range(b0, min(P, b0 + g.budget)))
        ctxs = [enc[int(idx[i])] + tokm[i, :T1].tolist() + [int(x[i])] for i in loc]
        g.patch.val_gen = None if val is None else val[loc]
        _, t = g.sample(ctxs, [int(idx[i]) for i in loc], T_MAIN[0], NSUITE, T1 + 1)
        tk[loc] = t
    g.patch.val_gen = None
    return tk


@torch.no_grad()
def rejouer_textes(g, enc, idx, textes, val=None):
    """Entropie aux indices t₀ … t₀ + 31 de textes de LONG jetons (indice i : loi du jeton t₀ + i) ; avec val (P, d), l'état
    caché de t₀ est remplacé à la couche g.patch.l."""
    base, head = g.model.model, g.model.get_output_embeddings()
    P = len(idx)
    S = np.zeros((P, NPOS), np.float32)
    nb = 24
    for b0 in range(0, P, nb):
        loc = list(range(b0, min(P, b0 + nb)))
        ids, attn, L = g.left_pad([enc[int(idx[i])] + textes[i].tolist() for i in loc])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        b = L - LONG                                 # position du jeton 0 de la suite (textes alignés à droite)
        if val is not None:
            g.patch.p, g.patch.val = b + T1, val[loc]
        h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
        g.patch.p, g.patch.val = None, None
        hk = h[:, b + T1 - 1:b + T1 - 1 + NPOS, :].reshape(len(loc) * NPOS, -1)
        flat = np.zeros(len(loc) * NPOS)
        for c0 in range(0, len(flat), 192):
            flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
        S[loc] = flat.reshape(len(loc), NPOS)
    return S


@torch.no_grad()
def capturer(g, enc, idx, tokm, x, couches):
    """État caché de la position t₀ (prompt + tokm[:, :t₀] + x) à la sortie de chaque couche de la grille : l → (P, d)."""
    P = len(idx)
    out = {l: None for l in couches}
    pris = {}

    def hk(i):
        def f(mod, args, o):
            if i in couches:
                pris[i] = sortie(o)[:, -1].detach().float().cpu()
        return f
    hs = [g.model.model.embed_tokens.register_forward_hook(hk(-1))]
    hs += [c.register_forward_hook(hk(i)) for i, c in enumerate(g.model.model.layers)]
    try:
        for b0 in range(0, P, g.budget):
            loc = list(range(b0, min(P, b0 + g.budget)))
            ids, attn, _ = g.left_pad([enc[int(idx[i])] + tokm[i, :T1].tolist() + [int(x[i])] for i in loc])
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            for l in couches:
                out[l] = pris[l] if out[l] is None else torch.cat([out[l], pris[l]])
    finally:
        for h in hs:
            h.remove()
    return out


@torch.no_grad()
def lentille(g, enc, idx, textes_t, textes_r):
    """Descriptif (idées de Kimi) : entropie de la lentille des logits head(norm(h_l)) rejoué − témoin aux k de KS_LENS, et
    rapport des normes ‖h_l(rejoué)‖ / ‖h_l(témoin)‖ sur k = 2–31, pour chaque couche."""
    base, head, norm = g.model.model, g.model.get_output_embeddings(), g.model.model.norm
    nl = len(base.layers)
    dS = np.zeros((nl, len(KS_LENS)))
    rn = np.zeros(nl)
    n_tot = 0
    pris = {}
    hs = [c.register_forward_hook((lambda i: lambda m_, a_, o: pris.__setitem__(i, sortie(o)))(i)) for i, c in enumerate(base.layers)]
    try:
        for b0 in range(0, len(idx), 8):
            loc = list(range(b0, min(len(idx), b0 + 8)))
            m = len(loc)
            seqs = [enc[int(idx[i])] + textes_t[i].tolist() for i in loc] + [enc[int(idx[i])] + textes_r[i].tolist() for i in loc]
            ids, attn, L = g.left_pad(seqs)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            q = L - LONG + T1 - 1                        # position qui prédit le jeton t₀
            for l in range(nl):
                H = pris[l].float()
                kk = [q + k for k in KS_LENS]
                z = head(norm(H[:, kk].to(norm.weight.device).to(norm.weight.dtype))).float()
                S = entropies_from_logits(z.reshape(-1, z.shape[-1]), [Q])[Q].reshape(2 * m, len(kk)).cpu().numpy()
                dS[l] += (S[m:] - S[:m]).sum(0)
                nt = H[:m, q + MID[0]:q + MID[1] + 1].norm(dim=-1)
                nr = H[m:, q + MID[0]:q + MID[1] + 1].norm(dim=-1)
                rn[l] += float((nr / nt.clamp(min=1e-6)).mean(1).sum())
            n_tot += m
            pris.clear()
    finally:
        for h in hs:
            h.remove()
    return dS / n_tot, rn / n_tot

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
W = slice(MID[0], MID[1] + 1)


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1y.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


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
        nl = len(model.model.layers)
        grille = grille_de(nl)
        t0g = time.time()
        res, info = run_main(g, enc, idx)
        print(f"[MESURE] {label} génération témoin + choc : {time.time() - t0g:.0f} s ; rang médian du choc "
              f"{np.median(info['rang']):.0f}, surprise médiane {np.median(info['surp']):.2f} nats ; {nl} couches, grille {grille}",
              flush=True)
        tokm, X = res["temoin"]["tok"], info["X"]
        own = tokm[:, T1]
        T_tem = tokm[:, :LONG]
        T_rej = T_tem.copy()
        T_rej[:, T1] = X

        # Descriptif : lentille des logits et normes, sur la moitié des prompts
        tl = time.time()
        moit = np.arange(0, P, 2)
        dS_lens, rn = lentille(g, enc, moit, T_tem[moit], T_rej[moit])
        print(f"[MESURE] {label} lentille des logits : {time.time() - tl:.0f} s", flush=True)
        for l in range(0, nl, 2 if nl > 8 else 1):
            print(f"[LENTILLE] {label} couche {l:>2} : rejoué − témoin, entropie de la lentille aux k {KS_LENS} : "
                  + " ".join(f"{x:+.3f}" for x in dS_lens[l]) + f" ; rapport des normes (k = 2–31) {rn[l]:.3f}")

        g.patch = Patcheur(model)
        ta = time.time()
        S_tem = rejouer_textes(g, enc, idx, T_tem)
        dd = np.abs(S_tem - res["temoin"]["S"][:, T1:T1 + NPOS])
        suite0 = generer(g, enc, idx, tokm, X)
        same0 = float(np.mean((suite0 == res["choc"]["tok"][:, T1 + 1:T1 + 1 + NSUITE]).all(1)))
        T_lib0 = np.concatenate([tokm[:, :T1], X[:, None], suite0], 1)
        S_rej0, S_lib0 = rejouer_textes(g, enc, idx, T_rej), rejouer_textes(g, enc, idx, T_lib0)
        moyw = lambda S: S[:, W].mean(1).astype(float)
        D0, I0 = moyw(S_rej0) - moyw(S_tem), moyw(S_lib0) - moyw(S_rej0)
        est = (time.time() - ta) * len(grille) * 2 * 0.7 / 60
        print(f"[VÉRIF] {label} témoin rejoué contre généré : écart moyen {dd.mean():.1e} ; suite libre du choc régénérée après t₀ "
              f"identique à la génération complète : {same0:.1%} des prompts")
        print(f"[PROGRESSION] {label} patching : estimation ~{est:.0f} min pour {2 * len(grille)} conditions", flush=True)
        if dd.mean() > 0.1:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
        if est / 60 > MAX_HOURS:
            raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        F0 = regle_F(D0, I0, boots, ok)
        print(f"[F] {label} sans crochet : D₀ = {F0['D']:+.4f} ; I₀ = {F0['I']:+.4f} IC95 [{F0['I_ci'][0]:+.4f} ; {F0['I_ci'][1]:+.4f}] "
              f"→ {F0['verdict']}" + (f" (A1u, mêmes prompts : D {A1U[label]['D']:+.4f}, I {A1U[label]['I']:+.4f})" if label in A1U else ""))
        amp = F0["verdict"] == "AMPLIFICATION"

        H_t = capturer(g, enc, idx, tokm, own, grille)
        H_c = capturer(g, enc, idx, tokm, X, grille)
        Dd, Id, Dn, In, sim = {}, {}, {}, {}, {}
        tp = time.time()
        for sens in ("deb", "bru"):
            for l in grille:
                g.patch.l = l
                if sens == "deb":                    # choc, état de t₀ restauré à la couche l
                    val = H_t[l]
                    suite = generer(g, enc, idx, tokm, X, val)
                    T_lib = np.concatenate([tokm[:, :T1], X[:, None], suite], 1)
                    S_r, S_l = rejouer_textes(g, enc, idx, T_rej, val), rejouer_textes(g, enc, idx, T_lib, val)
                    Dd[l], Id[l] = moyw(S_r) - moyw(S_tem), moyw(S_l) - moyw(S_r)
                    sim[("deb", l)] = float(np.mean((suite == tokm[:, T1 + 1:T1 + 1 + NSUITE]).all(1)))
                    if l == nl - 1:                  # dernière couche : la loi du jeton t₀ + 1 doit redevenir celle du témoin
                        k1_dern = float(np.abs(S_r[:, 1] - S_tem[:, 1]).mean())   # moyenne : le max subit le bruit fp16
                else:                                # témoin, état de t₀ choqué à la couche l
                    val = H_c[l]
                    suite = generer(g, enc, idx, tokm, own, val)
                    T_lib = np.concatenate([tokm[:, :T1 + 1], suite], 1)
                    S_r, S_l = rejouer_textes(g, enc, idx, T_tem, val), rejouer_textes(g, enc, idx, T_lib, val)
                    Dn[l], In[l] = moyw(S_r) - moyw(S_tem), moyw(S_l) - moyw(S_r)
                    sim[("bru", l)] = float(np.mean((suite == suite0).all(1)))
                if sens == "deb" and l == grille[0]:
                    print(f"[PROGRESSION] {label} patching : une condition en {time.time() - tp:.0f} s", flush=True)
        g.patch.l = None
        print(f"[MESURE] {label} patching : {time.time() - tp:.0f} s", flush=True)

        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} : {P} prompts, {nl} couches")
        c_deb = sim[("deb", -1)] >= 0.9 and abs(Dd[-1].mean()) < 0.002 and k1_dern < 5e-3
        c_bru = sim[("bru", -1)] >= 0.9
        print(f"[VÉRIF] {label} débruitage complet (l = −1) : suite identique au témoin {sim[('deb', -1)]:.1%}, D = "
              f"{Dd[-1].mean():+.5f}, à la dernière couche loi de t₀ + 1 rendue au témoin (écart moyen {k1_dern:.1e}) → "
              f"{'OK' if c_deb else 'ÉCHEC'} ; bruitage complet (l = −1) : suite identique au choc "
              f"{sim[('bru', -1)]:.1%}, I = {In[-1].mean():+.4f} (I₀ {I0.mean():+.4f}) → {'OK' if c_bru else 'ÉCHEC'}")
        ok_c = ok and c_deb and c_bru
        if ok and not ok_c:
            print(f"[VÉRIF] {label} : crochet défectueux, verdicts NON INTERPRÉTABLES")
        cim = lambda x: ci([x[ix].mean() for ix in boots])
        a = {"F0": F0, "grille": grille, "lentille": dS_lens.tolist(), "normes": rn.tolist(), "suite0_identique": same0,
             "controles": {"deb": c_deb, "bru": c_bru}, "courbes": {}}
        for l in grille:
            fd = ci([Id[l][ix].mean() / I0[ix].mean() for ix in boots])
            fn = ci([In[l][ix].mean() / I0[ix].mean() for ix in boots])
            a["courbes"][l] = {"f_deb": float(Id[l].mean() / I0.mean()), "f_deb_ci": fd, "f_bru": float(In[l].mean() / I0.mean()),
                               "f_bru_ci": fn, "D_deb": float(Dd[l].mean()), "D_bru": float(Dn[l].mean()),
                               "suite_deb_temoin": sim[("deb", l)], "suite_bru_choc": sim[("bru", l)]}
            print(f"[COURBE] {label} couche {l:>2} : f_déb {Id[l].mean() / I0.mean():+.2f} IC95 [{fd[0]:+.2f} ; {fd[1]:+.2f}] ; "
                  f"f_bru {In[l].mean() / I0.mean():+.2f} IC95 [{fn[0]:+.2f} ; {fn[1]:+.2f}] ; D_déb / D₀ {Dd[l].mean() / D0.mean():+.2f} ; "
                  f"D_bru / D₀ {Dn[l].mean() / D0.mean():+.2f} ; suites identiques : déb → témoin {sim[('deb', l)]:.0%}, "
                  f"bru → choc {sim[('bru', l)]:.0%}")
        y1 = regle_y1(Id, I0, grille, nl, boots, amp, ok_c)
        pos_l = [l for l in grille if l >= 0]
        y2 = regle_y2(Id, In, I0, pos_l, boots, amp, ok_c)
        print(f"[Y1] {label} profondeur de lecture : l₅₀ = {y1['l50']} IC95 [{y1['ci'][0]:.0f} ; {y1['ci'][1]:.0f}] (sur {nl} couches ; "
              f"{nl} = jamais atteint) ; f_déb à la dernière couche IC95 [{y1['f_derniere_ci'][0]:+.2f} ; {y1['f_derniere_ci'][1]:+.2f}] "
              f"→ {y1['verdict']}")
        print(f"[Y2] {label} additivité : A = moyenne de f_déb + f_bru sur les couches ≥ 0 = {y2['A']:.2f} IC95 [{y2['ci'][0]:.2f} ; "
              f"{y2['ci'][1]:.2f}] → {y2['verdict']}")
        a.update({"Y1": y1, "Y2": y2, "minutes": (time.time() - t_start) / 60})
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1y.npz"), I0=I0, D0=D0, grille=np.array(grille),
                            **{f"Id_{l}": Id[l] for l in grille}, **{f"In_{l}": In[l] for l in grille},
                            **{f"Dd_{l}": Dd[l] for l in grille}, **{f"Dn_{l}": Dn[l] for l in grille})
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min", flush=True)
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}", flush=True)
        ALL[label] = {"erreur": f"{type(e).__name__}: {str(e)[:300]}"}
    if g is not None and getattr(g, "patch", None) is not None:
        g.patch.remove()
    del model, tok, g
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ecrire_resume()

# %%
# ---------- Synthèse ----------
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1y — où l'amplification est-elle lue à partir du jeton choqué ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "Y1" in ALL[m]]
for m in labs:
    print(f"[SYNTHÈSE] {m:<18} : Y1 {ALL[m]['Y1']['verdict']} ; Y2 {ALL[m]['Y2']['verdict']}")
these = synthese(ALL[labs[0]]["Y1"]["verdict"], ALL[labs[0]]["Y2"]["verdict"], ok) if labs else "NON TESTABLE (modèle manquant)"
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « l'amplification est lue à partir du jeton choqué dans une bande de couches précise » : {these} "
      f"(prédiction SOUTENUE 30 %, RÉFUTÉE 40 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

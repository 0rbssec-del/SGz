# %% [markdown]
# # SGZ A1o — Le gabarit de conversation stabilise-t-il le modèle sur plusieurs tours, et améliore-t-il ses réponses ?
#
# **But.** A1j : sur un seul tour, le gabarit de conversation amortit nettement l'effet d'un jeton de choc sur l'entropie
# (part du gabarit ≈ 105 % de l'écart complet / brut, amortissement aussi en relatif). Thèse d'Erick : le gabarit serait
# un **stabilisateur** qui garde le modèle dans un bon régime **sur plusieurs tours**, et pas seulement au premier, au
# point d'améliorer ses réponses. A1o teste trois choses :
# 1. **Durabilité** : l'amortissement par le gabarit est-il le même quand le choc arrive au tour 6 qu'au tour 1 ?
# 2. **Contamination** : un choc au tour 1 laisse-t-il une trace d'entropie dans les tours 2–6, et le gabarit la contient-il ?
# 3. **Qualité** : sur une chaîne de 6 calculs vérifiables, le gabarit donne-t-il plus de réponses exactes ?
#
# **Deux formats, même modèle (Qwen/Qwen3-4B, comme A1j), mêmes conversations.**
#
# | Format | Ce que voit le modèle |
# |---|---|
# | brut | dialogue en texte simple : « Utilisateur : … » / « Assistant : … » (consigne présente, pas de jetons spéciaux) |
# | gabarit | gabarit de conversation officiel à chaque tour, réflexion désactivée |
#
# Le contexte est construit en ajoutant les jetons réellement générés (pas de re-tokenisation des réponses dans le
# volet S), avec l'ouverture de réponse vue à la génération (bloc de réflexion vide inclus) : c'est exactement ce que le
# modèle a lu en écrivant.
#
# **Volet S (stabilité).** 384 conversations de 6 tours ; à chaque tour, « Continue ce texte : » + un prompt différent
# (mêmes 768 prompts que A1c–A1j, une seule langue par conversation) ; réponse de 48 jetons, T = 0,6. Bras : témoin
# (6 tours) ; **choc1** = jeton de rang 1000–5000 forcé à la position 8 de la réponse du tour 1, conversation poursuivie
# jusqu'au tour 6 ; **choc3** et **choc6** = même choc au tour 3 ou 6, historique du témoin. Même bruit de Gumbel pour
# tous les bras d'une conversation.
#
# **Volet Q (qualité).** 768 chaînes de 6 calculs (« Prends le nombre a. Ajoute b. », puis « Multiplie le résultat
# par m », etc.), réponse « uniquement le résultat », 16 jetons au plus, T = 0,6 et T = 1,0. Exactitude **locale** :
# la réponse du tour t est juste si elle applique bien l'opération t au résultat que le modèle a donné au tour t − 1.
#
# ## Critères (fixés avant toute exécution)
#
# Effet direct du choc au tour j (passes rejouées, comme A1c–A1j) : **D_j** = moyenne sur k = 2–31 de S(choc) − S(témoin),
# S = entropie de Tsallis q = 1,45. Rapport entre formats **r_j** = D_j(gabarit) / D_j(brut), bootstrap apparié sur les
# conversations (B = 1000).
# - **V1 amortissement au tour 1** (réplication du sens d'A1j) : RÉDUIT si IC95(r_1) < 1 ; SANS EFFET si IC95 ⊂ [0,8 ; 1,2] ;
#   AUGMENTÉ si IC95 > 1 ; sinon INDÉTERMINÉ ; NON TESTABLE si l'IC95 de D_1(brut) contient 0.
# - **V2 durabilité** : a_j = −log r_j ; q = a_6 / a_1. DURABLE si borne basse de IC95(q) > 0,5 ; S'ESTOMPE si borne haute
#   < 0,5 ; sinon INDÉTERMINÉE. NON TESTABLE si V1 ≠ RÉDUIT ou si l'IC95 de D_6(brut) contient 0. (q au tour 3 : descriptif.)
# - **V3 contamination** : C = moyenne sur les tours 2–6 et sur les 48 pas de S(choc1) − S(témoin), génération libre.
#   Rapport gabarit / brut : CONTENUE si IC95 < 1 ; SANS EFFET si IC95 ⊂ [0,8 ; 1,2] ; AUGMENTÉE si IC95 > 1 ; sinon
#   INDÉTERMINÉE ; NON TESTABLE si l'IC95 de C(brut) contient 0.
# - **V4 qualité** (T = 0,6) : Δ = exactitude locale moyenne (gabarit − brut), moyenne sur les 6 tours, bootstrap apparié
#   sur les chaînes. AMÉLIORÉE si borne basse > +0,05 ; SANS GAIN si IC95 ⊂ [−0,05 ; +0,05] ; DÉGRADÉE si borne haute < 0 ;
#   sinon INDÉTERMINÉE. (Δ à T = 1,0 et interaction Δ(1,0) − Δ(0,6) : descriptifs.)
# - **Synthèse, thèse « stabilisateur durable qui améliore les réponses »** : SOUTENUE si V2 DURABLE et V4 AMÉLIORÉE
#   (et V3 ≠ AUGMENTÉE) ; RÉFUTÉE si V2 S'ESTOMPE, ou si V4 SANS GAIN ou DÉGRADÉE ; sinon INDÉTERMINÉE.
# - Témoins positifs synthétiques pour chaque règle ; s'ils échouent, tous les verdicts passent en NON TESTABLE.
#
# Écart avec le plan annoncé le 27 oct. : le bras « gabarit au premier tour seulement » est remplacé par le choc à
# différentes profondeurs (tours 1, 3, 6), car un gabarit limité au premier tour n'a pas de forme naturelle avec un modèle
# instruct ; la marge de qualité passe de 2 à 5 points, faute de puissance pour trancher « sans gain » à ±2 points.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1 RÉDUIT 75 %. V2 DURABLE 45 %, S'ESTOMPE 20 %, INDÉTERMINÉE ou NON TESTABLE 35 %. V3 contamination brut présente
# 60 %, CONTENUE 35 %. V4 AMÉLIORÉE 50 %, SANS GAIN 30 %. Synthèse SOUTENUE 25 %, RÉFUTÉE 40 %.
#
# ## Amendement du 27 oct. (essai 2), avant toute comparaison entre formats
#
# L'essai 1 a tourné le format brut en entier, puis s'est arrêté sur le format gabarit (mémoire GPU saturée : le
# pré-remplissage calculait les logits de toutes les positions). Aucun verdict n'a donc été calculé. Corrections, sans
# changer aucun critère ni aucun seuil :
# - **mémoire** : logits du dernier pas seulement, allocation CUDA par segments extensibles ;
# - **lecture des réponses (volet Q)** : le modèle écrit souvent le calcul (« 8 × 2 = 16. Réponse : 16 »), et la
#   lecture « premier entier » prenait l'opérande (8) au lieu du résultat ; elle prend désormais le nombre après
#   « Réponse / Résultat / Answer / Result », sinon après le dernier « = », sinon le premier entier ;
# - **longueur des réponses** : 16 → 40 jetons (16 coupait « Réponse : 16 » en « Réponse : 1 ») ; les jetons générés
#   sont enregistrés pour pouvoir relire les réponses.
# Les deux formats sont relancés en entier. Les mesures descriptives du format brut de l'essai 1 (volet S) ont été vues ;
# celles du format gabarit, jamais.
#
# ## Limites connues
#
# - Un seul modèle instruct, une seule formulation du gabarit et du dialogue brut.
# - En qualité, un gain du gabarit peut venir du simple **respect du format** appris à l'entraînement, et non d'une
#   stabilisation : la lecture du nombre reste permissive (voir l'amendement), mais le biais reste possible.
# - En format brut, Qwen3-4B écrit parfois un raisonnement en anglais (« Okay, the user wants… ») au lieu de répondre
#   (vu dans l'essai 1) : le format brut active une partie du mode « réflexion », que le gabarit coupe par un bloc vide.
# - Le contexte garde le bloc de réflexion vide de chaque tour (ajout pur) ; un usage réel l'efface des tours passés.
# - Le choc est en position 8 de la réponse (t₀ = 64 dans A1c–A1j) : les valeurs de D ne sont pas comparables à A1j,
#   seul le sens de V1 l'est.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (Qwen3-4B n'est pas restreint : aucun jeton Hugging Face requis).
# Durée estimée : ~1 h 30 à 2 h en tout [estimation].
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[GABARIT]`, `[VÉRIF]`, `[D]`, `[DURABILITÉ]`, `[CONTAMINATION]`,
# `[QUALITÉ]`, `[SYNTHÈSE]`.

# %%
import gc
import inspect
import json
import math
import os
import re
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # limite la fragmentation (essai 1 : OOM)
import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1o" if os.path.isdir("/kaggle/working") else "sgz_a1o_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_S = 0.6                                         # température du volet S (comme A1j)
T_Q = [0.6, 1.0]                                  # températures du volet Q
Q = 1.45
N_PROMPTS = 768                                   # réservoir de prompts (mêmes que A1c–A1j)
N_CONV, N_TOURS, L_REP, T0 = 384, 6, 48, 8
TOURS_CHOC = [1, 3, 6]
RANK_LO, RANK_HI = 1000, 5000
MID_D = (2, 31)
N_CHAINES, L_ARITH = 768, 40                      # 40 jetons : l'essai 1 (16) coupait « Réponse : 16 » en « 1 »
MARGE, MARGE_Q, SEUIL_Q = 0.2, 0.05, 0.5
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261030
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))    # par volet et par format
FORMATS = ["brut", "gabarit"]
MODELES_A_LANCER = ["Qwen/Qwen3-4B"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
CONSIGNE = {"fr": "Continue ce texte :\n\n{p}", "en": "Continue this text:\n\n{p}"}
ROLES = {"fr": ("Utilisateur : ", "Assistant : "), "en": ("User: ", "Assistant: ")}
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_CONV, L_REP, T0 = 12, 24, 4
    RANK_LO, RANK_HI = 20, 80
    MID_D = (2, 12)
    N_CHAINES, L_ARITH = 12, 8
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] volet S : {N_CONV} conversations × {N_TOURS} tours × {L_REP} jetons, choc en position {T0} aux tours "
      f"{TOURS_CHOC}, T = {T_S} ; volet Q : {N_CHAINES} chaînes, T = {T_Q} ; modèles={[m[0] for m in MODELS]}")
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
print(f"[PROMPTS] {len(PROMPTS)} prompts ({min(64, len(PROMPTS))} écrits à la main, {max(0, len(PROMPTS) - 64)} gabarits), mêmes que A1c–A1j")

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
    """Bruit de Gumbel par ligne, graine explicite par ligne (mêmes nombres pour tous les bras d'une conversation)."""
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


def contient0(c):
    return not (c[0] > 0 or c[1] < 0)


def rapport_verdict(x, y, boots, interpretable=True, nom_ref="référence"):
    """r = moyenne(x) / moyenne(y) (x : gabarit, y : brut), bootstrap apparié ; NON TESTABLE si y n'est pas ≠ 0."""
    r = float(x.mean() / y.mean())
    c = ci([x[ix].mean() / y[ix].mean() for ix in boots])
    cy = ci([y[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif contient0(cy):
        v = f"NON TESTABLE ({nom_ref} non significatif)"
    elif c[1] < 1:
        v = "RÉDUIT"
    elif c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
        v = "SANS EFFET"
    elif c[0] > 1:
        v = "AUGMENTÉ"
    else:
        v = "INDÉTERMINÉ"
    return {"r": r, "ci": c, "ci_ref": cy, "verdict": v}


def durabilite(Db, Dg, boots, interpretable=True):
    """V1 (r_1) et V2 (q = log r_6 / log r_1) ; Db, Dg : {tour: D par conversation}."""
    out = {f"r{j}": rapport_verdict(Dg[j], Db[j], boots, interpretable, f"D_{j}(brut)") for j in Db}
    v1 = out["r1"]["verdict"]
    out["V1"] = v1

    def q(j, ix):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(Dg[j][ix].mean() / Db[j][ix].mean()) / np.log(Dg[1][ix].mean() / Db[1][ix].mean())
    tout = np.arange(len(Db[1]))
    for j in [t for t in Db if t != 1]:
        out[f"q{j}"] = {"q": float(q(j, tout)), "ci": ci([q(j, ix) for ix in boots])}
    jl = max(Db)
    c = out[f"q{jl}"]["ci"]
    if not interpretable:
        v2 = "NON INTERPRÉTABLE"
    elif v1 != "RÉDUIT":
        v2 = "NON TESTABLE (pas d'amortissement au tour 1)"
    elif out[f"r{jl}"]["verdict"].startswith("NON TESTABLE"):
        v2 = f"NON TESTABLE (D_{jl}(brut) non significatif)"
    elif c[0] > SEUIL_Q:
        v2 = "DURABLE"
    elif c[1] < SEUIL_Q:
        v2 = "S'ESTOMPE"
    else:
        v2 = "INDÉTERMINÉE"
    out["V2"] = v2
    return out


def contamination(Cb, Cg, boots, interpretable=True):
    out = rapport_verdict(Cg, Cb, boots, interpretable, "C(brut)")
    out["C_brut"], out["C_gabarit"] = float(Cb.mean()), float(Cg.mean())
    out["ci_gabarit"] = ci([Cg[ix].mean() for ix in boots])
    out["V3"] = {"RÉDUIT": "CONTENUE", "AUGMENTÉ": "AUGMENTÉE", "INDÉTERMINÉ": "INDÉTERMINÉE"}.get(out["verdict"], out["verdict"])
    return out


def qualite(xg, xb, boots, interpretable=True):
    """Δ = exactitude moyenne gabarit − brut (par chaîne, moyenne sur les tours), bootstrap apparié."""
    d = xg - xb
    c = ci([d[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > MARGE_Q:
        v = "AMÉLIORÉE"
    elif c[0] >= -MARGE_Q and c[1] <= MARGE_Q:
        v = "SANS GAIN"
    elif c[1] < 0:
        v = "DÉGRADÉE"
    else:
        v = "INDÉTERMINÉE"
    return {"delta": float(d.mean()), "ci": c, "gabarit": float(xg.mean()), "brut": float(xb.mean()), "V4": v}


def synthese(v2, v3, v4, interpretable=True):
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if v2 == "DURABLE" and v4 == "AMÉLIORÉE" and v3 != "AUGMENTÉE":
        return "SOUTENUE"
    if v2 == "S'ESTOMPE" or v4 in ("SANS GAIN", "DÉGRADÉE"):
        return "RÉFUTÉE"
    return "INDÉTERMINÉE"


# Témoins positifs : effets connus, bruit multiplicatif par conversation
rng_pc = np.random.default_rng(SEED + 1)
Pp = 384
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
base_pc = rng_pc.lognormal(np.log(0.05), 0.4, Pp)
bruit = lambda f: f * base_pc * rng_pc.lognormal(0, 0.3, Pp) + rng_pc.normal(0, 0.005, Pp)


def synth_D(r1, r6):
    return {1: bruit(1.0), 6: bruit(1.0)}, {1: bruit(r1), 6: bruit(r6)}


checks = []
for nom, (r1, r6), cle, att in (("durable", (0.3, 0.3), "V2", "DURABLE"), ("s'estompe", (0.3, 0.85), "V2", "S'ESTOMPE"),
                                ("amortissement", (0.3, 0.3), "V1", "RÉDUIT"),
                                ("sans amortissement", (1.0, 1.0), "V2", "NON TESTABLE (pas d'amortissement au tour 1)")):
    checks.append((nom, durabilite(*synth_D(r1, r6), bt)[cle], att))
Cb_pc = bruit(1.0)
for nom, Cb, Cg, att in (("contamination contenue", Cb_pc, bruit(0.4), "CONTENUE"),
                         ("contamination égale", Cb_pc, bruit(1.0), "SANS EFFET"),
                         ("contamination absente", (lambda x: x - x.mean())(rng_pc.normal(0, 0.01, Pp)), bruit(1.0), "NON TESTABLE (C(brut) non significatif)")):
    checks.append((nom, contamination(Cb, Cg, bt)["V3"], att))
diff_pc = rng_pc.uniform(-0.25, 0.25, 768)                                 # difficulté partagée par les deux formats
bt_q = [rng_pc.integers(0, 768, 768) for _ in range(200)]
exa = lambda p: (rng_pc.random((768, N_TOURS)) < np.clip(p + diff_pc, 0, 1)[:, None]).mean(1)
for nom, pg, pb, att in (("qualité améliorée", 0.80, 0.60, "AMÉLIORÉE"), ("qualité égale", 0.70, 0.70, "SANS GAIN"),
                         ("qualité dégradée", 0.50, 0.70, "DÉGRADÉE")):
    checks.append((nom, qualite(exa(pg), exa(pb), bt_q)["V4"], att))
for nom, args, att in (("synthèse soutenue", ("DURABLE", "CONTENUE", "AMÉLIORÉE"), "SOUTENUE"),
                       ("synthèse réfutée", ("DURABLE", "CONTENUE", "SANS GAIN"), "RÉFUTÉE")):
    checks.append((nom, synthese(*args), att))
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<14} (attendu {att})")
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
# ---------- Génération (échantillonnage de Gumbel à graine explicite, comme A1c–A1j) ----------
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
        self.speciaux = set(self.forbid)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        npar = sum(p.numel() for p in model.parameters())
        self.budget = 48 if npar < 3e9 else 32
        self.t_first = None
        params = inspect.signature(model.forward).parameters                # logits du dernier pas seulement : le
        self.un_logit = ({"logits_to_keep": 1} if "logits_to_keep" in params else    # pré-remplissage d'un long contexte
                         {"num_logits_to_keep": 1} if "num_logits_to_keep" in params else {})   # ne matérialise pas B×L×V

    def left_pad(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.dev), attn.to(self.dev), L

    def masque(self, score, fin_libre):
        if not fin_libre:
            score[:, self.forbid] = -float("inf")
        if self.V > self.n_tok:
            score[:, self.n_tok:] = -float("inf")
        return score

    @torch.no_grad()
    def sample(self, ctxs, graine_de, T, nsteps, step0, force=None, fin_libre=False):
        n = len(ctxs)
        S, tk = np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int64)
        ids, attn, _ = self.left_pad(ctxs)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        past = None
        for s in range(nsteps):
            out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True,
                             **self.un_logit)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
            e1 = entropies_from_logits(z, [Q])[Q].cpu().numpy()
            seeds = [SEED + 100003 * graine_de[j] + step0 + s for j in range(n)]
            score = self.masque(z / T + gumbel_noise(self.V, seeds, z.device), fin_libre)
            nxt = torch.argmax(score, dim=-1)
            if force is not None:
                nxt = force(s, z, nxt)
            S[:, s], tk[:, s] = e1, nxt.cpu().numpy()
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = pos[:, -1:] + 1
        del past, out
        return S, tk

    def progress(self, t_lot, n_lots_total, quoi):
        if self.t_first is None:
            self.t_first = t_lot
            est = t_lot * n_lots_total / 60
            print(f"[PROGRESSION] {self.label} {quoi} : premier lot en {t_lot:.1f} s ; estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")

# %%
# ---------- Formats : segments de texte autour des messages ----------
def segments(tok, fmt, lang):
    """(pre, post, entre) : avant le 1er message de l'utilisateur ; après chaque message (ouverture de la réponse, telle
    que vue à la génération) ; entre la fin d'une réponse et le message suivant."""
    if fmt == "brut":
        u, a = ROLES[lang]
        return u, "\n" + a, "\n\n" + u
    if SMOKE_TEST or not hasattr(tok, "apply_chat_template"):
        return "<U>", "<A>", "<E><U>"
    msgs = [{"role": "user", "content": "§U1§"}, {"role": "assistant", "content": "§A1§"}, {"role": "user", "content": "§U2§"}]
    r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert r.count("§") == 6, "rendu du gabarit inattendu"
    iu1, ia1, iu2 = r.index("§U1§"), r.index("§A1§"), r.index("§U2§")
    return r[:iu1], r[iu2 + 4:], r[ia1 + 4:iu2]


def enc(tok, s):
    return tok.encode(s, add_special_tokens=False) if s else []


def contexte(seg, user_ids, t, hist):
    pre, post, entre = seg
    return hist + (pre if t == 1 else entre) + user_ids + post


def segs_tokens(tok, fmt):
    out = {}
    for lang in ("fr", "en"):
        pre, post, entre = segments(tok, fmt, lang)
        out[lang] = (enc(tok, pre), enc(tok, post), enc(tok, entre))
        print(f"[GABARIT] {fmt} {lang} : avant {pre!r} ; ouverture de réponse {post!r} ; entre deux tours {entre!r}")
    return out


# Conversations du volet S : une langue par conversation, 6 prompts distincts
POOL = {l: [i for i, x in enumerate(LANG) if x == l] for l in ("fr", "en")}
CONV = []
for c in range(N_CONV):
    lang = "fr" if c % 2 == 0 else "en"
    pool, m = POOL[lang], c // 2
    pas = len(pool) // N_TOURS
    CONV.append((lang, [PROMPTS[pool[(m + t * pas) % len(pool)]] for t in range(N_TOURS)]))
assert all(len(set(ps)) == N_TOURS for _, ps in CONV), "prompts répétés dans une conversation"
print(f"[CONVERSATIONS] {len(CONV)} conversations ({sum(l == 'fr' for l, _ in CONV)} en français) × {N_TOURS} tours")

# Chaînes du volet Q
OPS_TXT = {"fr": {"debut": "Prends le nombre {a}. Ajoute {b}. Réponds uniquement par le résultat, en chiffres.",
                  "add": "Ajoute {b} au résultat. Réponds uniquement par le résultat, en chiffres.",
                  "sub": "Soustrais {b} du résultat. Réponds uniquement par le résultat, en chiffres.",
                  "mul": "Multiplie le résultat par {b}. Réponds uniquement par le résultat, en chiffres."},
           "en": {"debut": "Take the number {a}. Add {b}. Answer with the result only, in digits.",
                  "add": "Add {b} to the result. Answer with the result only, in digits.",
                  "sub": "Subtract {b} from the result. Answer with the result only, in digits.",
                  "mul": "Multiply the result by {b}. Answer with the result only, in digits."}}


def appliquer(k, b, v):
    return v + b if k in ("debut", "add") else (v - b if k == "sub" else v * b)


def chaines(n):
    rng = np.random.default_rng(SEED + 21)
    out = []
    for c in range(n):
        a = int(rng.integers(10, 61))
        v, ops = a, []
        for t in range(N_TOURS):
            if t == 0:
                k, b = "debut", int(rng.integers(2, 31))
            else:
                k = ("add", "sub", "mul")[int(rng.integers(3))]
                b = int(rng.integers(2, 4)) if k == "mul" else int(rng.integers(2, 31))
                if k == "mul" and v * b > 2000:
                    k, b = "sub", int(rng.integers(2, 31))
                if k == "sub" and v - b < 1:
                    k = "add"
            ops.append((k, b))
            v = appliquer(k, b, v)
        out.append({"lang": "fr" if c % 2 == 0 else "en", "a": a, "ops": ops})
    return out


CHAINES = chaines(N_CHAINES)
NUM = re.compile(r"-?\d+")
CLE = re.compile(r"(?:r[ée]ponse|r[ée]sultat|answer|result)\s*(?:est|is)?\s*[:=]?\s*\**\s*(-?\d+)", re.I)


def lire_nombre(txt):
    """Nombre donné comme résultat : après « Réponse / Résultat / Answer / Result » s'il y en a un, sinon après le
    dernier « = », sinon le premier entier (le modèle écrit souvent le calcul : « 8 × 2 = 16. Réponse : 16 »)."""
    t = txt.replace(" ", "").replace(" ", "")
    t = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", t)
    m = list(CLE.finditer(t))
    if m:
        return int(m[-1].group(1))
    if "=" in t:
        n = NUM.search(t.rsplit("=", 1)[1])
        if n:
            return int(n.group())
    n = NUM.search(t)
    return int(n.group()) if n else None


for txt, att in (("45", 45), ("Résultat : 1 234.", 1234), ("The answer is -7", -7), ("rien", None), ("**88**", 88),
                 ("8 × 2 = 16. Réponse : 16", 16), ("1 + 18 = 19", 19), ("The result is **42**.", 42),
                 ("85 - 3 = 82. Answer: 82", 82), ("59 + 26 = 85", 85)):
    assert lire_nombre(txt) == att, (txt, lire_nombre(txt))

# %%
# ---------- Volet S : stabilité sur 6 tours ----------
ARMS_TOURS = ([("temoin", t) for t in range(1, N_TOURS + 1)] + [("choc1", t) for t in range(1, N_TOURS + 1)]
              + [(f"choc{j}", j) for j in TOURS_CHOC if j != 1])


@torch.no_grad()
def run_S(g, segs):
    tok = g.tok
    P = len(CONV)
    rng = np.random.default_rng(SEED + 5)
    rang_force = {j: rng.integers(RANK_LO, RANK_HI + 1, P) for j in TOURS_CHOC}
    users = [[enc(tok, CONSIGNE[l].format(p=p)) for p in ps] for l, ps in CONV]
    S = {k: np.zeros((P, L_REP), np.float32) for k in ARMS_TOURS}
    TK = {k: np.zeros((P, L_REP), np.int64) for k in ARMS_TOURS}
    CTX = {j: [None] * P for j in TOURS_CHOC}
    RG = {j: np.zeros(P, np.int64) for j in TOURS_CHOC}
    nc = max(1, g.budget // 3)
    n_lots = math.ceil(P / nc)
    for b0 in range(0, P, nc):
        tl = time.time()
        rows = list(range(b0, min(P, b0 + nc)))
        n = len(rows)
        h_t, h_c = [[] for _ in rows], [[] for _ in rows]
        for t in range(1, N_TOURS + 1):
            c_t = [contexte(segs[CONV[r][0]], users[r][t - 1], t, h_t[i]) for i, r in enumerate(rows)]
            c_c = [contexte(segs[CONV[r][0]], users[r][t - 1], t, h_c[i]) for i, r in enumerate(rows)]
            extra = t in TOURS_CHOC and t != 1
            ctxs = c_t + c_c + (c_t if extra else [])
            choc = slice(n, 2 * n) if t == 1 else (slice(2 * n, 3 * n) if extra else None)

            def force(s, z, nxt, choc=choc, t=t, rows=rows, n=n):
                if choc is None:
                    return nxt
                if s < T0:
                    nxt[choc] = nxt[:n]                                     # appariement exact avant le choc
                elif s == T0:
                    zz = g.masque(z[choc].clone(), False)
                    order = torch.argsort(zz, dim=-1, descending=True)
                    for j, r in enumerate(rows):
                        rk = int(min(rang_force[t][r], order.shape[1] - 2))
                        if int(order[j, rk]) == int(nxt[j]):               # jamais le jeton du témoin
                            rk += 1
                        nxt[choc.start + j] = order[j, rk]
                        RG[t][r] = rk
                return nxt

            Sx, tk = g.sample(ctxs, rows * (len(ctxs) // n), T_S, L_REP, 1009 * t, force)
            S[("temoin", t)][rows], TK[("temoin", t)][rows] = Sx[:n], tk[:n]
            S[("choc1", t)][rows], TK[("choc1", t)][rows] = Sx[n:2 * n], tk[n:2 * n]
            if extra:
                S[(f"choc{t}", t)][rows], TK[(f"choc{t}", t)][rows] = Sx[2 * n:], tk[2 * n:]
            if t in TOURS_CHOC:
                for i, r in enumerate(rows):
                    CTX[t][r] = c_t[i]
            for i in range(n):
                h_t[i] = c_t[i] + tk[i].tolist()
                h_c[i] = c_c[i] + tk[n + i].tolist()
        g.progress(time.time() - tl, n_lots, "volet S")
    for j in TOURS_CHOC:
        ref, fo = TK[("temoin", j)], TK[(f"choc{j}", j)]
        same = np.array_equal(ref[:, :T0], fo[:, :T0])
        chg = float(np.mean(ref[:, T0] != fo[:, T0]))
        print(f"[VÉRIF] {g.label} choc au tour {j} : identique avant le choc : {same} ; jeton changé : {chg:.0%} ; "
              f"rang médian {int(np.median(RG[j]))} ; contexte médian {int(np.median([len(c) for c in CTX[j]]))} jetons", flush=True)
        if not same or chg < 0.99:
            raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
    return S, TK, CTX


@torch.no_grad()
def rejouer(g, ctxs, conts):
    """Entropie de chaque pas d'une réponse imposée (forçage par l'enseignant), en une passe sans cache."""
    base, head = g.model.model, g.model.get_output_embeddings()
    n, Kp = len(ctxs), len(conts[0])
    ids, attn, L = g.left_pad([c + s[:-1] for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    hk = h[:, L - Kp:, :].reshape(n * Kp, -1)
    flat = np.zeros(n * Kp, np.float64)
    for c0 in range(0, n * Kp, 192):
        flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), [Q])[Q].cpu().numpy()
    return flat.reshape(n, Kp).astype(np.float32)


@torch.no_grad()
def run_replay_S(g, S, TK, CTX):
    P = len(CONV)
    SR = {}
    for j in TOURS_CHOC:
        arm = f"choc{j}"
        SR[("temoin", j)], SR[(arm, j)] = np.zeros((P, L_REP), np.float32), np.zeros((P, L_REP), np.float32)
        for b0 in range(0, P, 8):
            rows = list(range(b0, min(P, b0 + 8)))
            ct = [TK[("temoin", j)][r].tolist() for r in rows]
            cf = [c[:T0] + [int(TK[(arm, j)][r, T0])] + c[T0 + 1:] for c, r in zip(ct, rows)]
            out = rejouer(g, [CTX[j][r] for r in rows] * 2, ct + cf)
            SR[("temoin", j)][rows], SR[(arm, j)][rows] = out[:len(rows)], out[len(rows):]
        d_all = np.abs(SR[("temoin", j)] - S[("temoin", j)])
        d1 = np.abs(SR[(arm, j)][:, T0 + 1] - S[(arm, j)][:, T0 + 1])
        print(f"[VÉRIF] {g.label} tour {j} rejoué contre généré : témoin écart moyen {d_all.mean():.1e}, max {d_all.max():.1e} ; "
              f"choc à k=1 {d1.mean():.1e}", flush=True)
        if d_all.mean() > 0.1 or d1.mean() > 0.1:
            raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
    return SR


def mesures_S(S, SR):
    """Par conversation : D_j, S0_j, I_j (tours de choc) et contamination C (tours 2–6 après le choc du tour 1)."""
    m = {"D": {}, "S0": {}, "I": {}, "courbe_D": {}}
    for j in TOURS_CHOC:
        arm = f"choc{j}"
        Rrej = (SR[(arm, j)] - SR[("temoin", j)])[:, T0 + 1:].astype(float)
        Rlib = (S[(arm, j)] - S[("temoin", j)])[:, T0 + 1:].astype(float)
        m["D"][j] = win(Rrej, MID_D)
        m["S0"][j] = win(SR[("temoin", j)][:, T0 + 1:].astype(float), MID_D)
        m["I"][j] = win(Rlib, MID_D) - m["D"][j]
        m["courbe_D"][j] = [float(Rrej[:, a - 1:b].mean()) for a, b in ((1, 1), (2, 7), (8, 15), (16, 31)) if b <= Rrej.shape[1]]
    par_tour = np.stack([(S[("choc1", t)] - S[("temoin", t)]).astype(float).mean(1) for t in range(2, N_TOURS + 1)], 1)
    m["C"] = par_tour.mean(1)
    m["C_tours"] = par_tour.mean(0).tolist()
    return m

# %%
# ---------- Volet Q : chaînes de calcul sur 6 tours ----------
@torch.no_grad()
def run_Q(g, segs, fmt, T):
    tok = g.tok
    P = len(CHAINES)
    loc, absol, lu = np.zeros((P, N_TOURS)), np.zeros((P, N_TOURS)), np.zeros((P, N_TOURS))
    S4 = np.zeros((P, N_TOURS))
    tks = np.zeros((P, N_TOURS, L_ARITH), np.int64)                            # pour relire les réponses après coup
    extrait = None
    nb = g.budget
    n_lots = math.ceil(P / nb)
    for b0 in range(0, P, nb):
        tl = time.time()
        rows = list(range(b0, min(P, b0 + nb)))
        hist = [[] for _ in rows]
        prev_mod = [CHAINES[r]["a"] for r in rows]
        prev_vrai = list(prev_mod)
        for t in range(1, N_TOURS + 1):
            ctxs = []
            for i, r in enumerate(rows):
                ch = CHAINES[r]
                k, b = ch["ops"][t - 1]
                ctxs.append(contexte(segs[ch["lang"]], enc(tok, OPS_TXT[ch["lang"]][k].format(a=ch["a"], b=b)), t, hist[i]))
            Sx, tk = g.sample(ctxs, [10 ** 6 + r for r in rows], T, L_ARITH, 50000 + 1009 * t, None, fin_libre=True)
            for i, r in enumerate(rows):
                ids = tk[i].tolist()
                cut = next((p for p, x in enumerate(ids) if x in g.speciaux), len(ids))
                txt = tok.decode(ids[:cut])
                if fmt == "brut":
                    ligne = next((x for x in txt.strip().split("\n") if x.strip()), "")
                    rep = enc(tok, ligne)
                else:
                    ligne, rep = txt, ids[:cut]
                x = lire_nombre(ligne)
                k, b = CHAINES[r]["ops"][t - 1]
                att_loc, vrai = appliquer(k, b, prev_mod[i]), appliquer(k, b, prev_vrai[i])
                loc[r, t - 1], absol[r, t - 1], lu[r, t - 1] = x == att_loc, x == vrai, x is not None
                S4[r, t - 1], tks[r, t - 1] = Sx[i, :4].mean(), tk[i]
                prev_vrai[i], prev_mod[i] = vrai, (x if x is not None else att_loc)
                hist[i] = ctxs[i] + rep
                if r == 0 and t == N_TOURS:
                    extrait = tok.decode(hist[i])
        g.progress(time.time() - tl, n_lots, f"volet Q T={T}")
    return {"local": loc, "absolu": absol, "lu": lu, "S4": S4, "tok": tks, "extrait": extrait}

# %%
# ---------- Boucle principale : un modèle, deux formats, deux volets ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {"formats": {}}
MS, MQ = {}, {}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1o.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def charger(ids, fp32=False):
    for mid in ids:
        print(f"[CHARGEMENT] {label} : téléchargement / chargement de {mid}…", flush=True)
        try:
            model, tok, mode = load_model(mid, fp32=fp32)
            print(f"[CHARGEMENT] {label} : {mid} en {mode}")
            if getattr(tok, "pad_token_id", None) is None:
                tok.pad_token = tok.eos_token
            return model, tok
        except Exception as e:
            print(f"[CHARGEMENT] échec {mid} : {type(e).__name__}: {str(e)[:300]}")
    raise RuntimeError("modèle non chargé : rien à mesurer")


def volets(model, tok, fmt):
    segs = segs_tokens(tok, fmt)
    g = Gen(model, tok, f"{label}|{fmt}")
    S, TK, CTX = run_S(g, segs)
    SR = run_replay_S(g, S, TK, CTX)
    q = {}
    for T in T_Q:
        q[T] = run_Q(Gen(model, tok, f"{label}|{fmt}"), segs, fmt, T)
    return S, TK, CTX, SR, q


label, ids = MODELS[0]
t_all = time.time()
model, tok = charger(ids)
fp32 = False
for fmt in FORMATS:
    t_start = time.time()
    try:
        try:
            S, TK, CTX, SR, q = volets(model, tok, fmt)
        except FloatingPointError as e:
            print(f"[VÉRIF] {fmt} : {e} en fp16 → rechargement en fp32 pour ce format et les suivants", flush=True)
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model, tok = charger(ids, fp32=True)
            fp32 = True
            S, TK, CTX, SR, q = volets(model, tok, fmt)
        MS[fmt], MQ[fmt] = mesures_S(S, SR), q
        m = MS[fmt]
        a = {"D": {j: float(m["D"][j].mean()) for j in TOURS_CHOC}, "S0": {j: float(m["S0"][j].mean()) for j in TOURS_CHOC},
             "I": {j: float(m["I"][j].mean()) for j in TOURS_CHOC}, "courbe_D": m["courbe_D"],
             "C": float(m["C"].mean()), "C_tours": m["C_tours"],
             "Q": {T: {"local": float(q[T]["local"].mean()), "absolu": float(q[T]["absolu"].mean()),
                       "lu": float(q[T]["lu"].mean()), "local_par_tour": q[T]["local"].mean(0).tolist(),
                       "S4_par_tour": q[T]["S4"].mean(0).tolist()} for T in T_Q},
             "fp32": fp32, "minutes": (time.time() - t_start) / 60}
        ALL["formats"][fmt] = a
        for j in TOURS_CHOC:
            print(f"[D] {fmt:<8} tour {j} : D = {a['D'][j]:+.4f} ; S0 = {a['S0'][j]:.3f} ; I = {a['I'][j]:+.4f} ; "
                  f"D par tranches de k (1 | 2–7 | 8–15 | 16–31) : " + " ".join(f"{v:+.3f}" for v in a["courbe_D"][j]))
        print(f"[CONTAMINATION] {fmt:<8} : C (tours 2–6) = {a['C']:+.4f} ; par tour : "
              + " ".join(f"{v:+.4f}" for v in a["C_tours"]))
        for T in T_Q:
            r = a["Q"][T]
            print(f"[QUALITÉ] {fmt:<8} T={T} : exactitude locale {r['local']:.1%} (absolue {r['absolu']:.1%}, nombre lu "
                  f"{r['lu']:.1%}) ; par tour : " + " ".join(f"{v:.0%}" for v in r["local_par_tour"]))
        print(f"[EXTRAIT] {fmt} conversation 0, tour 1, choc : {tok.decode(TK[('choc1', 1)][0].tolist())!r}")
        print(f"[EXTRAIT] {fmt} chaîne 0 (T={T_Q[0]}) : {q[T_Q[0]]['extrait'][-600:]!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_{fmt}_a1o.npz"),
                            **{f"S_{arm}_t{t}": S[(arm, t)] for arm, t in ARMS_TOURS},
                            **{f"tok_{arm}_t{t}": TK[(arm, t)] for arm, t in ARMS_TOURS},
                            **{f"Srej_{arm}_t{j}": SR[(arm, j)] for arm, j in SR},
                            **{f"Q_{k}_T{T}": q[T][k] for T in T_Q for k in ("local", "absolu", "lu", "S4", "tok")})
        print(f"[DURÉE] {fmt} : {a['minutes']:.1f} min", flush=True)
    except Exception as e:
        print(f"[ERREUR] {fmt} : {type(e).__name__}: {str(e)[:500]}", flush=True)
        ALL["formats"][fmt] = {"erreur": f"{type(e).__name__}: {str(e)[:300]}"}
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ecrire_resume()

# %%
# ---------- Contrastes entre formats et verdicts ----------
print(f"\n{'=' * 100}\n[RÉSULTATS] {label} : gabarit contre brut (mêmes conversations, bootstrap apparié)")
if all(f in MS for f in FORMATS):
    P = len(CONV)
    rngB = np.random.default_rng(SEED + 11)
    boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
    dur = durabilite(MS["brut"]["D"], MS["gabarit"]["D"], boots, ok)
    for j in TOURS_CHOC:
        r = dur[f"r{j}"]
        print(f"[DURABILITÉ] tour {j} : r = D(gabarit)/D(brut) = {r['r']:.2f} IC95 [{r['ci'][0]:.2f} ; {r['ci'][1]:.2f}] "
              f"{r['verdict']}" + (f" ; q = a_{j}/a_1 = {dur[f'q{j}']['q']:.2f} IC95 [{dur[f'q{j}']['ci'][0]:.2f} ; "
                                   f"{dur[f'q{j}']['ci'][1]:.2f}]" if j != 1 else ""))
    dr = {j: rapport_verdict(MS["gabarit"]["D"][j] / MS["gabarit"]["S0"][j].mean(),
                             MS["brut"]["D"][j] / MS["brut"]["S0"][j].mean(), boots, ok) for j in TOURS_CHOC}
    print("[DURABILITÉ] relatif Dr = D/S0 (descriptif) : " + " ; ".join(f"tour {j} ×{dr[j]['r']:.2f}" for j in TOURS_CHOC))
    print(f"[DURABILITÉ] V1 amortissement au tour 1 : {dur['V1']} (prédiction RÉDUIT 75 %)")
    print(f"[DURABILITÉ] V2 durabilité : {dur['V2']} (prédiction DURABLE 45 %, S'ESTOMPE 20 %)")
    con = contamination(MS["brut"]["C"], MS["gabarit"]["C"], boots, ok)
    print(f"[CONTAMINATION] C brut = {con['C_brut']:+.4f} IC95 [{con['ci_ref'][0]:+.4f} ; {con['ci_ref'][1]:+.4f}] ; "
          f"C gabarit = {con['C_gabarit']:+.4f} IC95 [{con['ci_gabarit'][0]:+.4f} ; {con['ci_gabarit'][1]:+.4f}] ; "
          f"rapport {con['r']:.2f} IC95 [{con['ci'][0]:.2f} ; {con['ci'][1]:.2f}]")
    print(f"[CONTAMINATION] V3 : {con['V3']} (prédiction CONTENUE 35 %)")
else:
    dur, con, dr = {"V1": "NON TESTABLE (format manquant)", "V2": "NON TESTABLE (format manquant)"}, \
        {"V3": "NON TESTABLE (format manquant)"}, {}
    print("[DURABILITÉ] NON TESTABLE (format manquant)")
if all(f in MQ for f in FORMATS):
    P = len(CHAINES)
    rngQ = np.random.default_rng(SEED + 13)
    bootsQ = [rngQ.integers(0, P, P) for _ in range(BOOT_N)]
    qual = {T: qualite(MQ["gabarit"][T]["local"].mean(1), MQ["brut"][T]["local"].mean(1), bootsQ, ok) for T in T_Q}
    for T in T_Q:
        r = qual[T]
        print(f"[QUALITÉ] T={T} : exactitude locale gabarit {r['gabarit']:.1%}, brut {r['brut']:.1%} ; Δ = {r['delta']:+.3f} "
              f"IC95 [{r['ci'][0]:+.3f} ; {r['ci'][1]:+.3f}] : {r['V4']}" + ("" if T == T_Q[0] else " (descriptif)"))
    dd = lambda ix: ((MQ["gabarit"][T_Q[1]]["local"].mean(1) - MQ["brut"][T_Q[1]]["local"].mean(1))[ix].mean()
                     - (MQ["gabarit"][T_Q[0]]["local"].mean(1) - MQ["brut"][T_Q[0]]["local"].mean(1))[ix].mean())
    inter = {"valeur": float(dd(np.arange(P))), "ci": ci([dd(ix) for ix in bootsQ])}
    print(f"[QUALITÉ] interaction Δ(T={T_Q[1]}) − Δ(T={T_Q[0]}) = {inter['valeur']:+.3f} IC95 [{inter['ci'][0]:+.3f} ; "
          f"{inter['ci'][1]:+.3f}] (descriptif : > 0 si le gabarit compense davantage une température haute)")
    print(f"[QUALITÉ] V4 (T={T_Q[0]}) : {qual[T_Q[0]]['V4']} (prédiction AMÉLIORÉE 50 %, SANS GAIN 30 %)")
else:
    qual, inter = {T_Q[0]: {"V4": "NON TESTABLE (format manquant)"}}, {}
    print("[QUALITÉ] NON TESTABLE (format manquant)")
these = synthese(dur["V2"], con["V3"], qual[T_Q[0]]["V4"], ok)
ALL.update({"durabilite": dur, "relatif": dr, "contamination": con, "qualite": qual, "interaction_T": inter, "these": these})
print(f"[SYNTHÈSE] thèse « le gabarit est un stabilisateur durable qui améliore les réponses » : {these} "
      f"(V2 {dur['V2']} ; V3 {con['V3']} ; V4 {qual[T_Q[0]]['V4']}) — prédiction SOUTENUE 25 %, RÉFUTÉE 40 %")
ALL["minutes"] = (time.time() - t_all) / 60
print(f"[DURÉE] totale : {ALL['minutes']:.1f} min")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

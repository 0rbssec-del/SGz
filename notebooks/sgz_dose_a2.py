# %% [markdown]
# # SGZ A2 — Des doses répétées de corruption maintiennent-elles le modèle au bord de la bascule ?
#
# **Thèse d'Erick (24 oct. 2026).** « Si le modèle se stabilise dans sa corruption, l'ajout de corruption à petite dose
# le maintient dans un état suffisamment corrompu pour rester à la limite du bord du chaos. »
#
# **Version testée.** La corruption du contexte est un second bouton de réglage du régime, côté entrée. Avec des doses
# répétées, l'entropie se stabilise à un niveau fixé par la dose ; il existe une dose seuil au-delà de laquelle le modèle
# bascule (passage soudain à une entropie haute, vu en v3 vers T ≈ 1,1) ; juste sous ce seuil, les fluctuations sont
# maximales et l'état reste stable. Alternatives : réponse graduelle sans bord (bouton de réglage ordinaire) ; cliquet
# (les doses s'accumulent et l'entropie dérive) ; corruption = simple température effective.
#
# **Dispositif.** Qwen3-1.7B-Base, les 256 prompts techniques de A1c (volet E), T = 0,8, 448 jetons. Une dose tous les
# 32 jetons, du pas 64 au pas 416 (12 doses) : on corrompt les 32 derniers jetons générés, chaque mot avec la probabilité
# p, le type tiré au hasard parmi les cinq de A1c (points au lieu des espaces, lettres inversées, chiffre à la place
# d'une lettre, mélange français-anglais, faute d'orthographe). Doses p = 0 (aller-retour décodage → encodage seul),
# 0,03 ; 0,1 ; 0,2 ; 0,35 ; 0,5 ; 0,75. Bras température sans corruption : T = 0,8 ; 0,9 ; 1,0 ; 1,1 ; 1,2. Volet
# additivité : une dose unique au pas 64 (160 jetons, comme A1c) : aller-retour, chaque type seul, les cinq combinés.
# Hasard commun entre bras (même bruit de Gumbel par prompt et par pas). Nouvelle graine.
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# - Moitiés de la période de dosage : H1 = pas 64–255, H2 = pas 256–447. Entropie S = S₁,₄₅(p₁).
# - **Bascule** : la moyenne glissante de S sur 32 pas dépasse 1,6 entre les pas 64 et 447 (seuil calé sur A1b : 0 % des
#   séquences à T = 0,8, 89 % à T = 1,1). f(p) = part des séquences qui basculent.
# - **Fluctuation G1 hors chocs** : moyenne de |S_t − S_{t−1}| sur H2, en ne gardant que la seconde moitié de chaque bloc
#   de 32 pas (les 16 pas qui suivent chaque dose sont exclus, pour ne pas compter le saut que la dose produit).
# - **Dérive** : D(p) = [S̄(H2) − S̄(H1)] à la dose p, moins la même quantité à p = 0.
# - SEUIL entre deux doses voisines p_i < p_{i+1} : f(p_i) ≤ 0,10 et f(p_{i+1}) ≥ 0,50.
# - PIC à une dose intermédiaire p (0,03 à 0,5) : IC95(G1(p) − G1(0)) > 0 et IC95(G1(p) − G1(0,75)) > 0.
# - STATIONNAIRE à p : IC95(D(p)) ⊂ [−0,05 ; +0,05]. CLIQUET à p : IC95(D(p)) > 0.
# - **Verdict de la thèse** : SOUTENUE s'il existe i avec SEUIL(p_i, p_{i+1}), PIC(p_i) et STATIONNAIRE(p_i).
#   RÉFUTÉE si aucun SEUIL et aucun PIC (réponse graduelle sans bord), ou si CLIQUET à toutes les doses p > 0.
#   Sinon INDÉTERMINÉE.
# - **Additivité** (dose unique) : E = effet moyen sur k = 1–16 contre l'aller-retour ; ρ = E(combinée) / Σ E(types seuls).
#   ADDITIF si IC95(ρ) ⊂ [0,7 ; 1,3] ; sinon SOUS-ADDITIF si IC95(ρ) < 1 ; SUR-ADDITIF si IC95(ρ) > 1 ; sinon INDÉTERMINÉ.
# - **Corruption contre température** (descriptif) : pour chaque dose, température équivalente T_eq (même S̄(H2),
#   interpolation entre les bras température) ; ÉQUIVALENT À UNE TEMPÉRATURE si, pour toutes les doses dans la plage,
#   |G1 − G1(T_eq)| ≤ 20 % et |f − f(T_eq)| ≤ 0,10 ; sinon DIFFÉRENT.
# - NON INTERPRÉTABLE en SMOKE_TEST ; NON TESTABLE si le témoin positif échoue ; arrêt si la vérification du cache échoue
#   deux fois (le notebook bascule d'abord sur un recalcul complet du contexte à chaque dose).
#
# **Puissance (d'après A1b et A1c).** G1 par prompt : écart-type ≈ 0,13 → erreur type d'une différence appariée
# ≈ 0,01 avec 256 prompts. Part de bascules : erreur type ≤ 3 points. Additivité : effets seuls de 0,14 à 0,20 (A1c),
# IC de chacun ≈ ±0,03.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Additivité : SOUS-ADDITIF 65 % (S est bornée, saturation), ADDITIF 25 %, SUR-ADDITIF 10 %. SEUIL présent 30 % ; PIC à
# une dose intermédiaire 30 % ; CLIQUET à toutes les doses 25 %. Thèse : SOUTENUE 10 %, RÉFUTÉE 45 %, INDÉTERMINÉE 45 %.
# Corruption contre température : DIFFÉRENT 60 %.
#
# ## Limites connues
#
# - Une seule température de base (0,8) et un seul modèle ; la grille de doses peut manquer un seuil très étroit.
# - Le mélange des types fixe leurs proportions ; la dose réalisée (part des mots réellement changés) est rapportée.
# - Le seuil de bascule (1,6) est calé sur des séquences de 288 jetons ; ici elles en ont 448.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : 35 à 50 min [estimation]. Budget : 3 h.
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[PROGRESSION]`, `[ADD]`, `[DOSE]`, `[SEUIL]`, `[THÈSE]`.

# %%
import difflib
import gc
import inspect
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a2" if os.path.isdir("/kaggle/working") else "sgz_a2_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_BASE = 0.8
T_ARMS = [0.8, 0.9, 1.0, 1.1, 1.2]
DOSES = [0.0, 0.03, 0.1, 0.2, 0.35, 0.5, 0.75]
Q = 1.45
N_PROMPTS, N_SEL, GEN_LEN, T0, SPAN, GEN_ADD = 768, 256, 448, 64, 32, 160
BASCULE_S, P_MOT = 1.6, 0.5
EARLY, LATE_ADD = (1, 16), (64, 95)
MARGE_D, MARGE_ADD, MARGE_T = 0.05, 0.3, 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261026
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
CORR_TYPES = ["points", "inversion", "chiffres", "melange", "fautes"]
ADD_ARMS = ["identite"] + CORR_TYPES + ["combinee"]
MODELS = [("Qwen3-1.7B-Base", ["Qwen/Qwen3-1.7B-Base"])]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, N_SEL, GEN_LEN, T0, SPAN, GEN_ADD = 16, 8, 112, 16, 8, 56
    EARLY, LATE_ADD = (1, 4), (24, 39)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]
DOSE_STEPS = list(range(T0, GEN_LEN - SPAN + 1, SPAN))            # 64, 96, …, 416
MID = T0 + (GEN_LEN - T0) // 2                                     # début de H2 (256)
H1, H2 = (T0, MID - 1), (MID, GEN_LEN - 1)
W_B = SPAN
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_BASE} doses={DOSES} ({len(DOSE_STEPS)} doses, pas {DOSE_STEPS[0]}–{DOSE_STEPS[-1]}, tous les {SPAN}) ; "
      f"bras température {T_ARMS} ; prompts={N_SEL} ; longueur={GEN_LEN} ; modèles={[m[0] for m in MODELS]}")
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
if SMOKE_TEST:
    SEL = list(range(N_SEL))
else:                                      # les 256 prompts techniques de A1c (16 thèmes × 8 débuts × 2 langues)
    SEL = [i for i, (_, _, tech, ti) in enumerate(ALL_P) if tech and ti < 16][:N_SEL]
assert len(SEL) == N_SEL
PROMPTS_S = [PROMPTS[i] for i in SEL]
LANG_S = [LANG[i] for i in SEL]
print(f"[PROMPTS] {len(PROMPTS_S)} prompts techniques ({sum(l == 'fr' for l in LANG_S)} en français), identiques au volet E de A1c")

# %%
# ---------- Corruptions du texte et détecteurs de style ----------
RE_WORD = re.compile(r"[^\W\d_]+")
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8", "g": "9", "z": "2", "é": "3", "è": "3", "à": "4"}
ACC = str.maketrans("éèêëàâäîïôöùûüçÉÈÊÀÇ", "eeeeaaaiioouuucEEEAC")
FR2EN = {"le": "the", "la": "the", "les": "the", "un": "a", "une": "a", "des": "some", "de": "of", "du": "of the",
         "et": "and", "est": "is", "sont": "are", "dans": "in", "pour": "for", "avec": "with", "sur": "on", "par": "by",
         "qui": "which", "que": "that", "ce": "this", "cette": "this", "il": "it", "elle": "she", "ils": "they",
         "nous": "we", "pas": "not", "plus": "more", "mais": "but", "ou": "or", "comme": "like", "aussi": "also",
         "très": "very", "entre": "between", "sans": "without", "sous": "under", "être": "be", "avoir": "have",
         "a": "has", "été": "been", "peut": "can", "deux": "two", "premier": "first", "temps": "time", "monde": "world",
         "ville": "city", "eau": "water", "leur": "their", "leurs": "their", "au": "to the", "aux": "to the", "en": "in",
         "lorsque": "when", "quand": "when", "tous": "all", "ces": "these", "donc": "therefore", "selon": "according to"}
EN2FR = {"the": "le", "a": "un", "an": "un", "of": "de", "and": "et", "is": "est", "are": "sont", "in": "dans",
         "for": "pour", "with": "avec", "on": "sur", "by": "par", "which": "qui", "that": "que", "this": "ce", "it": "il",
         "they": "ils", "we": "nous", "not": "pas", "more": "plus", "but": "mais", "or": "ou", "like": "comme",
         "also": "aussi", "very": "très", "between": "entre", "without": "sans", "under": "sous", "be": "être",
         "have": "avoir", "has": "a", "been": "été", "can": "peut", "two": "deux", "first": "premier", "time": "temps",
         "world": "monde", "city": "ville", "water": "eau", "their": "leur", "his": "son", "her": "sa", "its": "son",
         "to": "à", "from": "de", "was": "était", "were": "étaient", "will": "va", "when": "quand", "there": "là",
         "all": "tous", "one": "un", "these": "ces", "into": "dans", "than": "que", "such": "tel", "as": "comme"}
# Mots marqueurs d'une langue, sans les mots ambigus entre les deux (a, on, plus, son, but, or, premier, as…)
EN_MARK = set(EN2FR) - {"a", "on", "as", "son", "but", "or", "an"}
FR_MARK = set(FR2EN) - {"a", "plus", "premier", "son", "en", "ce", "ou"}


def _pick(cands, rng):
    """Mots à corrompre : chacun avec la probabilité P_MOT, au moins un si possible."""
    sel = [m for m in cands if rng.random() < P_MOT]
    if not sel and cands:
        sel = [cands[int(rng.integers(len(cands)))]]
    return {id(m) for m in sel}


def _inverse(w, rng):
    c = [i for i in range(1, len(w) - 2) if w[i] != w[i + 1]]     # lettres intérieures, première et dernière gardées
    if not c:
        return w
    i = c[int(rng.integers(len(c)))]
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _chiffre(w, rng):
    c = [i for i, ch in enumerate(w) if ch.lower() in LEET]
    if not c:
        return w
    i = c[int(rng.integers(len(c)))]
    return w[:i] + LEET[w[i].lower()] + w[i + 1:]


def _faute(w, rng):
    ops = (["accent"] if w.translate(ACC) != w else []) + ["supprime", "double"]
    op = ops[int(rng.integers(len(ops)))]
    if op == "accent":
        return w.translate(ACC)
    i = int(rng.integers(1, len(w) - 1))
    return w[:i] + w[i + 1:] if op == "supprime" else w[:i] + w[i] + w[i:]


def _traduit(w, lang):
    d = FR2EN if lang == "fr" else EN2FR
    t = d.get(w.lower(), w)
    return t[:1].upper() + t[1:] if w[:1].isupper() and t else t


def corrupt(text, kind, lang, rng):
    if kind == "identite":
        return text
    if kind == "points":                                   # points au lieu des espaces entre les mots
        return re.sub(r"(?<=\S) (?=\S)", ".", text)
    words = list(RE_WORD.finditer(text))
    if kind == "melange":
        d = FR2EN if lang == "fr" else EN2FR
        cands = [m for m in words if m.group().lower() in d]
        f = lambda w: _traduit(w, lang)
    else:
        cands = [m for m in words if len(m.group()) >= 4]
        f = {"inversion": lambda w: _inverse(w, rng), "chiffres": lambda w: _chiffre(w, rng),
             "fautes": lambda w: _faute(w, rng)}[kind]
    sel = _pick(cands, rng)
    out, last = [], 0
    for m in words:
        if id(m) in sel:
            out += [text[last:m.start()], f(m.group())]
            last = m.end()
    return "".join(out) + text[last:]


def taux_style(text, kind, lang, lex):
    """Détecteur propre à chaque corruption (plus haut = plus de ce style)."""
    if kind == "points":
        return 100.0 * len(re.findall(r"[^\W\d_]\.[^\W\d_]", text)) / max(1, len(text))
    if kind == "chiffres":
        return 100.0 * len(re.findall(r"[^\W\d_]\d|\d[^\W\d_]", text)) / max(1, len(text))
    ws = [w.lower() for w in RE_WORD.findall(text)]
    if kind == "melange":
        mark = EN_MARK if lang == "fr" else FR_MARK
        return float(np.mean([w in mark for w in ws])) if ws else 0.0
    ws = [w for w in ws if len(w) >= 4]                    # inversion, fautes : mots hors lexique des textes témoins
    return float(np.mean([w not in lex for w in ws])) if ws else 0.0


def change_rate(a, b):
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
TYPES_MIX = CORR_TYPES


def corrupt_mix(text, lang, p, rng):
    """Dose p : chaque mot est corrompu avec la probabilité p, type tiré au hasard parmi les cinq.
    Renvoie (texte, part des mots réellement changés)."""
    words = list(RE_WORD.finditer(text))
    d = FR2EN if lang == "fr" else EN2FR
    out, last, changed = [], 0, 0
    for m in words:
        if rng.random() >= p:
            continue
        kind = TYPES_MIX[int(rng.integers(len(TYPES_MIX)))]
        w = m.group()
        if kind == "points":                                   # l'espace qui suit le mot devient un point
            e = m.end()
            if e + 1 < len(text) and text[e] == " " and not text[e + 1].isspace():
                out += [text[last:e], "."]
                last, changed = e + 1, changed + 1
            continue
        if kind == "melange":
            nw = _traduit(w, lang) if w.lower() in d else w
        elif len(w) < 4:
            nw = w
        else:
            nw = {"inversion": _inverse, "chiffres": _chiffre, "fautes": _faute}[kind](w, rng)
        if nw != w:
            out += [text[last:m.start()], nw]
            last, changed = m.end(), changed + 1
    return "".join(out) + text[last:], changed / max(1, len(words))


def corrupt_combined(text, lang, rng):
    """Les cinq corruptions de A1c appliquées l'une après l'autre au même segment (mélange en premier, points en dernier)."""
    for kind in ("melange", "inversion", "chiffres", "fautes", "points"):
        text = corrupt(text, kind, lang, rng)
    return text

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


def seg_mean(S, a, b):
    return S[:, a:b + 1].mean(1)


def bascule(S):
    """Vrai si la moyenne glissante de S sur W_B pas dépasse BASCULE_S entre T0 et la fin."""
    x = S[:, T0:]
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0))), 1)
    r = (c[:, W_B:] - c[:, :-W_B]) / W_B
    return r.max(1) > BASCULE_S


KEEP_G1 = np.array([t for t in range(H2[0], H2[1] + 1) if (t - T0) % SPAN > SPAN // 2])   # t et t − 1 hors chocs


def g1_hors_chocs(S):
    return np.abs(S[:, KEEP_G1] - S[:, KEEP_G1 - 1]).mean(1)


def derive(S):
    return seg_mean(S, *H2) - seg_mean(S, *H1)


def verdict_seuil(f):
    """Indices i tels que f(p_i) ≤ 0,10 et f(p_{i+1}) ≥ 0,50."""
    return [i for i in range(len(f) - 1) if f[i] <= 0.10 and f[i + 1] >= 0.50]


def diff_ci(x, y, boots):
    d = x - y
    return float(d.mean()), ci([d[ix].mean() for ix in boots])


def ratio_add(comb, singles, boots, interpretable=True):
    tot = np.sum(singles, axis=0)
    r = float(comb.mean() / tot.mean())
    c = ci([comb[ix].mean() / tot[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] >= 1 - MARGE_ADD and c[1] <= 1 + MARGE_ADD:
        v = "ADDITIF"
    elif c[1] < 1:
        v = "SOUS-ADDITIF"
    elif c[0] > 1:
        v = "SUR-ADDITIF"
    else:
        v = "INDÉTERMINÉ"
    return {"rho": r, "ci": c, "somme": float(tot.mean()), "combinee": float(comb.mean()), "verdict": v}


def these_verdict(f, pic, stat, cliq):
    """f : parts de bascule par dose ; pic, stat, cliq : booléens par dose (pic seulement pour les doses intermédiaires)."""
    seuils = verdict_seuil(f)
    if any(pic[i] and stat[i] for i in seuils):
        return "SOUTENUE", seuils
    if (not seuils and not any(pic[1:-1])) or all(cliq[1:]):
        return "RÉFUTÉE", seuils
    return "INDÉTERMINÉE", seuils


# Témoin positif : règles de décision sur cas connus, et corruption par dose sur un texte technique
rng_pc = np.random.default_rng(SEED + 1)
bt = [rng_pc.integers(0, 256, 256) for _ in range(200)]
base_pc = rng_pc.normal(0.15, 0.05, 256)
checks = [
    ("seuil net", these_verdict([0.0, 0.0, 0.05, 0.6, 0.9, 1, 1], [0, 0, 1, 0, 0, 0, 0], [1] * 7, [0] * 7)[0], "SOUTENUE"),
    ("réponse graduelle", these_verdict([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], [0] * 7, [1] * 7, [0] * 7)[0], "RÉFUTÉE"),
    ("cliquet partout", these_verdict([0.0, 0.0, 0.05, 0.6, 0.9, 1, 1], [0, 0, 1, 0, 0, 0, 0], [0] * 7, [0] + [1] * 6)[0], "RÉFUTÉE"),
    ("seuil sans pic", these_verdict([0.0, 0.0, 0.05, 0.6, 0.9, 1, 1], [0] * 7, [1] * 7, [0] * 7)[0], "INDÉTERMINÉE"),
    ("additif", ratio_add(5 * base_pc + rng_pc.normal(0, 0.05, 256), [base_pc] * 5, bt)["verdict"], "ADDITIF"),
    ("sous-additif", ratio_add(2 * base_pc + rng_pc.normal(0, 0.05, 256), [base_pc] * 5, bt)["verdict"], "SOUS-ADDITIF"),
    ("sur-additif", ratio_add(8 * base_pc + rng_pc.normal(0, 0.05, 256), [base_pc] * 5, bt)["verdict"], "SUR-ADDITIF"),
]
SAMPLE = ("La chromodynamique quantique décrit l'interaction forte entre les quarks et les gluons, dont la liberté "
          "asymptotique explique le comportement des hadrons à très haute énergie.")
real = [np.mean([corrupt_mix(SAMPLE, "fr", p, np.random.default_rng([SEED, i]))[1] for i in range(50)]) for p in DOSES]
checks.append(("dose 0 = texte intact", "OUI" if corrupt_mix(SAMPLE, "fr", 0.0, np.random.default_rng(0))[0] == SAMPLE else "NON", "OUI"))
checks.append(("dose réalisée croissante", "OUI" if all(np.diff(real) > 0) else "NON", "OUI"))
for name, got, exp in checks:
    print(f"[TÉMOIN POSITIF] {name:<24} : {got:<13} (attendu {exp})")
print("[TÉMOIN POSITIF] dose nominale → part des mots changés sur le texte témoin : "
      + " ; ".join(f"{p:g} → {r:.2f}" for p, r in zip(DOSES, real)))
for p in (0.1, 0.35, 0.75):
    print(f"[EXEMPLE] dose {p:<4} : {corrupt_mix(SAMPLE, 'fr', p, np.random.default_rng([SEED, 7]))[0][:120]}")
print(f"[EXEMPLE] combinée  : {corrupt_combined(SAMPLE, 'fr', np.random.default_rng([SEED, 7]))[:120]}")
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


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == "smoke-qwen3-random":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
        torch.manual_seed(SEED)
        return Qwen3ForCausalLM(cfg).eval().to(DEVICE), SmokeTokenizer(512), "fp32 aléatoire (smoke)"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    else:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": 0}, "fp16 1 GPU"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda":
        assert next(model.parameters()).dtype == torch.float16, "le modèle n'est pas en fp16"
    return model, tok, mode

# %%
# ---------- Génération avec doses : le segment corrompu remplace les 32 derniers jetons dans le cache ----------
class CacheMismatch(RuntimeError):
    pass


def crop_cache(past, n_remove):
    cur = past.get_seq_length()
    try:
        past.crop(-n_remove)                   # entier négatif : retirer n jetons (anciennes et nouvelles versions)
    except Exception as e:
        raise CacheMismatch(f"cache : crop impossible ({type(e).__name__}: {e})")
    if past.get_seq_length() != cur - n_remove:
        raise CacheMismatch(f"cache : longueur {past.get_seq_length()} au lieu de {cur - n_remove}")


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
        small = sum(p.numel() for p in model.parameters()) < 3e9
        self.budget = 64 if small else 16
        self.mode = os.environ.get("CACHE_MODE", "cache")   # « cache » ou « recalcul »
        self.check_pending, self.check_log = True, None
        self.t_first = None

    def left_pad(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.dev), attn.to(self.dev), L

    def fwd(self, ids, attn, pos, past):
        return self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True,
                          logits_to_keep=1)

    @torch.no_grad()
    def run(self, enc_rows, prompt_of, T, nsteps, dose_steps, corr_fn):
        """Génère nsteps jetons ; aux pas de dose_steps, corr_fn(texte, j, s) remplace les SPAN derniers jetons générés.
        Renvoie S, surprisal, jetons tirés (alignés sur le pas) et la part des mots changés à chaque dose."""
        n = len(enc_rows)
        S, sur, tk = (np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int32))
        hist = [[] for _ in range(n)]
        realized = []
        ids, attn, _ = self.left_pad(enc_rows)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        nreal = [len(e) for e in enc_rows]
        past = None
        for s in range(nsteps):
            if corr_fn is not None and s in dose_steps:
                fr = []
                new = []
                for j in range(n):
                    txt, r = corr_fn(self.tok.decode(hist[j][-SPAN:]), j, s)
                    fr.append(r)
                    new.append(self.tok.encode(txt, add_special_tokens=False) or hist[j][-1:])
                realized.append(float(np.mean(fr)))
                start = [nreal[j] - SPAN for j in range(n)]          # position du premier jeton du segment remplacé
                for j in range(n):
                    hist[j] = hist[j][:-SPAN] + new[j]
                    nreal[j] = start[j] + len(new[j])
                if self.mode == "cache":
                    crop_cache(past, SPAN - 1)                         # le dernier jeton tiré n'est pas encore en cache
                    attn = attn[:, :past.get_seq_length()]
                    m = max(len(x) for x in new)
                    chunk = torch.full((n, m), self.pad, dtype=torch.long)
                    cm = torch.zeros((n, m), dtype=torch.long)
                    cp = torch.zeros((n, m), dtype=torch.long)
                    for j, x in enumerate(new):
                        chunk[j, m - len(x):] = torch.tensor(x)
                        cm[j, m - len(x):] = 1
                        cp[j, m - len(x):] = torch.arange(start[j], start[j] + len(x))
                    ids, pos = chunk.to(self.dev), cp.to(self.dev)
                    attn = torch.cat([attn, cm.to(self.dev)], dim=1)
                else:                                                  # recalcul complet du contexte
                    past = None
                    ids, attn, _ = self.left_pad([enc_rows[j] + hist[j] for j in range(n)])
                    pos = (attn.cumsum(-1) - 1).clamp(min=0)
            out = self.fwd(ids, attn, pos, past)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
            e1 = entropies_from_logits(z, [Q])[Q]
            if self.check_pending and corr_fn is not None and s in dose_steps and self.mode == "cache":
                fids, fattn, _ = self.left_pad([enc_rows[j] + hist[j] for j in range(n)])
                fpos = (fattn.cumsum(-1) - 1).clamp(min=0)
                zf = self.fwd(fids, fattn, fpos, None).logits[:, -1, :].float()
                d = (entropies_from_logits(zf, [Q])[Q] - e1).abs()
                self.check_pending, self.check_log = False, (float(d.mean()), float(d.max()))
                if float(d.mean()) > 0.01:
                    raise CacheMismatch(f"cache contre recalcul : écart d'entropie moyen {float(d.mean()):.3f}")
            seeds = [SEED + 100003 * prompt_of[j] + s for j in range(n)]
            score = z / T + gumbel_noise(self.V, seeds, z.device)
            score[:, self.forbid] = -float("inf")
            if self.V > self.n_tok:
                score[:, self.n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            sur[:, s] = (-torch.log_softmax(z.double(), -1).gather(1, nxt[:, None])[:, 0]).cpu().numpy()
            S[:, s], tk[:, s] = e1.cpu().numpy(), nxt.cpu().numpy()
            for j in range(n):
                hist[j].append(int(nxt[j]))
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = torch.tensor([[nreal[j]] for j in range(n)], device=self.dev)
            nreal = [x + 1 for x in nreal]
        del past, out
        return S, sur, tk, realized

    def progress(self, t_lot, n_lots_total):
        if self.t_first is None:
            self.t_first = t_lot
            est = t_lot * n_lots_total / 60
            print(f"[PROGRESSION] {self.label} : premier lot en {t_lot:.1f} s ; estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")


def run_arm(g, enc, T, nsteps, dose_steps, make_fn, n_lots_total):
    P = len(enc)
    S, sur, tk = (np.zeros((P, nsteps), np.float32), np.zeros((P, nsteps), np.float32), np.zeros((P, nsteps), np.int32))
    real = []
    for b0 in range(0, P, g.budget):
        tl = time.time()
        rows = list(range(b0, min(P, b0 + g.budget)))
        fn = make_fn(rows) if make_fn is not None else None
        Sb, sb, tb, rb = g.run([enc[r] for r in rows], [SEL[r] for r in rows], T, nsteps, set(dose_steps), fn)
        S[rows], sur[rows], tk[rows] = Sb, sb, tb
        real.append(rb)
        g.progress(time.time() - tl, n_lots_total)
    return S, sur, tk, (np.mean(real, axis=0).tolist() if real and real[0] else [])


def mix_fn(p):
    def make(rows):
        def fn(txt, j, s):
            r = rows[j]
            return corrupt_mix(txt, LANG_S[r], p, np.random.default_rng([SEED, SEL[r], s, int(round(p * 1000))]))
        return fn
    return make


def single_fn(kind):
    def make(rows):
        def fn(txt, j, s):
            r = rows[j]
            rng = np.random.default_rng([SEED, SEL[r], s, 7 + ADD_ARMS.index(kind)])
            if kind == "combinee":
                return corrupt_combined(txt, LANG_S[r], rng), float("nan")
            return corrupt(txt, kind, LANG_S[r], rng), float("nan")
        return fn
    return make


def run_all(g, enc):
    lots = math.ceil(len(enc) / g.budget)
    n_lots = lots * (len(DOSES) + len(T_ARMS) + len(ADD_ARMS) * GEN_ADD / GEN_LEN)
    res = {}
    t0 = time.time()
    for T in T_ARMS:                                                   # bras température, sans corruption
        res[("T", T)] = run_arm(g, enc, T, GEN_LEN, [], None, n_lots)
        print(f"[MESURE] {g.label} bras T={T} : {time.time() - t0:.0f} s cumulés", flush=True)
    for p in DOSES:                                                    # p = 0 : aller-retour seul à chaque dose
        res[("dose", p)] = run_arm(g, enc, T_BASE, GEN_LEN, DOSE_STEPS, mix_fn(p), n_lots)
        if p == DOSES[0]:
            print(f"[VÉRIF] {g.label} mode du cache : {g.mode} ; cache contre recalcul complet au premier pas après une dose : "
                  f"écart d'entropie moyen {g.check_log[0]:.1e}, max {g.check_log[1]:.1e}" if g.check_log else
                  f"[VÉRIF] {g.label} mode du cache : {g.mode}", flush=True)
        print(f"[MESURE] {g.label} dose {p} : {time.time() - t0:.0f} s cumulés ; part des mots changés par dose "
              f"{np.mean(res[('dose', p)][3]) if res[('dose', p)][3] else 0:.3f}", flush=True)
    for a in ADD_ARMS:                                                 # additivité : une dose unique au pas T0
        res[("add", a)] = run_arm(g, enc, T_BASE, GEN_ADD, [T0], single_fn(a), n_lots)
    print(f"[MESURE] {g.label} volet additivité : {time.time() - t0:.0f} s cumulés", flush=True)
    return res

# %%
# ---------- Analyse ----------
def analyse(res, tok, interpretable):
    ok = interpretable and POSITIVE_CONTROL_OK
    rng = np.random.default_rng(SEED + 3)
    P = res[("dose", 0.0)][0].shape[0]
    boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]
    out = {"doses": {}, "temperatures": {}, "additivite": {}}
    S0 = res[("dose", 0.0)][0].astype(float)
    g0, gmax, d0 = g1_hors_chocs(S0), g1_hors_chocs(res[("dose", DOSES[-1])][0].astype(float)), derive(S0)
    f, pic, stat, cliq = [], [], [], []
    for p in DOSES:
        S = res[("dose", p)][0].astype(float)
        b, g, D = bascule(S), g1_hors_chocs(S), derive(S) - d0
        m_g0, c_g0 = diff_ci(g, g0, boots)
        m_gm, c_gm = diff_ci(g, gmax, boots)
        cD = ci([D[ix].mean() for ix in boots])
        f.append(float(b.mean()))
        pic.append(bool(c_g0[0] > 0 and c_gm[0] > 0))
        stat.append(bool(cD[0] >= -MARGE_D and cD[1] <= MARGE_D))
        cliq.append(bool(cD[0] > 0))
        b1 = bascule(S[:, :MID])
        out["doses"][p] = {"f": f[-1], "ci_f": ci([b[ix].mean() for ix in boots]), "f_H1": float(b1.mean()),
                           "niveau_H2": float(seg_mean(S, *H2).mean()), "niveau_H1": float(seg_mean(S, *H1).mean()),
                           "G1": float(g.mean()), "G1_moins_G1_0": [m_g0, c_g0], "G1_moins_G1_max": [m_gm, c_gm],
                           "derive": float(D.mean()), "ci_derive": cD, "pic": pic[-1], "stationnaire": stat[-1],
                           "cliquet": cliq[-1], "ecart_type_niveau": float(seg_mean(S, *H2).std()),
                           "dose_realisee": float(np.mean(res[("dose", p)][3])) if res[("dose", p)][3] else 0.0}
    for T in T_ARMS:
        S = res[("T", T)][0].astype(float)
        out["temperatures"][T] = {"f": float(bascule(S).mean()), "niveau_H2": float(seg_mean(S, *H2).mean()),
                                  "G1": float(g1_hors_chocs(S).mean())}
    if not ok:
        these, seuils = ("NON INTERPRÉTABLE" if not interpretable else "NON TESTABLE (témoin positif raté)"), verdict_seuil(f)
    else:
        these, seuils = these_verdict(f, pic, stat, cliq)
    out["these"], out["seuils"] = these, [[DOSES[i], DOSES[i + 1]] for i in seuils]
    # Corruption contre température (descriptif)
    lv = np.array([out["temperatures"][T]["niveau_H2"] for T in T_ARMS])
    order = np.argsort(lv)
    comp, eq = {}, True
    for p in DOSES[1:]:
        L = out["doses"][p]["niveau_H2"]
        if not (lv.min() <= L <= lv.max()):
            comp[p] = "hors plage"
            continue
        Teq = float(np.interp(L, lv[order], np.array(T_ARMS)[order]))
        gT = float(np.interp(Teq, T_ARMS, [out["temperatures"][T]["G1"] for T in T_ARMS]))
        fT = float(np.interp(Teq, T_ARMS, [out["temperatures"][T]["f"] for T in T_ARMS]))
        same = abs(out["doses"][p]["G1"] - gT) <= MARGE_T * gT and abs(out["doses"][p]["f"] - fT) <= 0.10
        eq &= same
        comp[p] = {"T_eq": Teq, "G1_T": gT, "f_T": fT, "meme": bool(same)}
    out["contre_temperature"] = {"par_dose": comp, "verdict": "ÉQUIVALENT À UNE TEMPÉRATURE" if eq else "DIFFÉRENT"}
    # Additivité
    Sid = res[("add", "identite")][0].astype(float)
    E = {a: (res[("add", a)][0].astype(float) - Sid)[:, T0:T0 + EARLY[1]].mean(1) for a in ADD_ARMS[1:]}
    El = {a: (res[("add", a)][0].astype(float) - Sid)[:, T0 + LATE_ADD[0] - 1:T0 + LATE_ADD[1]].mean(1) for a in ADD_ARMS[1:]}
    out["additivite"] = ratio_add(E["combinee"], [E[a] for a in CORR_TYPES], boots, ok)
    out["additivite"]["seuls"] = {a: float(E[a].mean()) for a in CORR_TYPES}
    out["additivite"]["tardif"] = {a: float(El[a].mean()) for a in ADD_ARMS[1:]}
    # Contagion du style dans le dernier bloc (descriptif)
    lex = set()
    for r in range(P):
        lex |= {w.lower() for w in RE_WORD.findall(tok.decode(res[("T", T_BASE)][2][r].tolist()))}
    for p in DOSES:
        tk = res[("dose", p)][2]
        txt = [tok.decode(tk[r, GEN_LEN - SPAN:].tolist()) for r in range(P)]
        out["doses"][p]["style_fin"] = {k: float(np.mean([taux_style(t, k, LANG_S[r], lex) for r, t in enumerate(txt)]))
                                        for k in ("points", "chiffres", "melange", "fautes")}
    # Identité contre bras T = 0,8 sans aller-retour
    tid, tpl = res[("dose", 0.0)][2], res[("T", T_BASE)][2]
    out["verif_identite"] = {"jetons_identiques": float((tid == tpl).mean()),
                             "ecart_niveau_H2": float(abs(seg_mean(S0, *H2).mean() - seg_mean(res[("T", T_BASE)][0].astype(float), *H2).mean()))}
    return out

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ALL = {}
for label, ids in MODELS:
    t_start = time.time()
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
        ALL[label] = {"these": "NON TESTABLE (modèle non chargé)"}
        continue
    try:
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        g = Gen(model, tok, label)
        enc = [tok.encode(p)[:12] if SMOKE_TEST else tok.encode(p) for p in PROMPTS_S]
        try:
            res = run_all(g, enc)
        except CacheMismatch as e:
            print(f"[VÉRIF] {label} : {e} → reprise en mode « recalcul » (contexte recalculé à chaque dose)", flush=True)
            g.mode, g.t_first = "recalcul", None
            res = run_all(g, enc)
        a = analyse(res, tok, interpretable)
        a["minutes"] = (time.time() - t_start) / 60
        a["mode_cache"] = g.mode
        ALL[label] = a
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label}")
        v = a["verif_identite"]
        print(f"[VÉRIF] {label} aller-retour seul (dose 0) contre T = {T_BASE} sans aller-retour : jetons identiques "
              f"{v['jetons_identiques']:.1%}, écart de niveau H2 {v['ecart_niveau_H2']:.3f}")
        r = a["additivite"]
        print(f"[ADD] {label} : {r['verdict']} — combinée {r['combinee']:+.4f} contre somme des types seuls {r['somme']:+.4f} ; "
              f"ρ = {r['rho']:.2f} IC95 [{r['ci'][0]:.2f} ; {r['ci'][1]:.2f}] ; seuls : "
              + ", ".join(f"{k} {x:+.3f}" for k, x in r["seuls"].items()) + " ; tardif (k=64–95) combinée "
              f"{r['tardif']['combinee']:+.3f}")
        for T, t in a["temperatures"].items():
            print(f"[TEMP] {label} T={T} : niveau H2 {t['niveau_H2']:.3f} ; G1 hors chocs {t['G1']:.3f} ; bascules {t['f']:.1%}")
        for p, d in a["doses"].items():
            print(f"[DOSE] {label} p={p:<4} (mots changés {d['dose_realisee']:.2f}) : niveau H1 {d['niveau_H1']:.3f} → H2 "
                  f"{d['niveau_H2']:.3f} ; bascules {d['f']:.1%} IC95 [{d['ci_f'][0]:.1%} ; {d['ci_f'][1]:.1%}] (H1 seule "
                  f"{d['f_H1']:.1%}) ; G1 {d['G1']:.3f} (− dose 0 : {d['G1_moins_G1_0'][0]:+.3f} IC95 "
                  f"[{d['G1_moins_G1_0'][1][0]:+.3f} ; {d['G1_moins_G1_0'][1][1]:+.3f}] ; − dose max : {d['G1_moins_G1_max'][0]:+.3f} "
                  f"IC95 [{d['G1_moins_G1_max'][1][0]:+.3f} ; {d['G1_moins_G1_max'][1][1]:+.3f}]) ; dérive {d['derive']:+.3f} "
                  f"IC95 [{d['ci_derive'][0]:+.3f} ; {d['ci_derive'][1]:+.3f}] ; PIC {d['pic']} STATIONNAIRE {d['stationnaire']} "
                  f"CLIQUET {d['cliquet']}")
            print(f"[STYLE] {label} p={p:<4} dernier bloc : " + " ; ".join(f"{k} {x:.3f}" for k, x in d["style_fin"].items()))
        print(f"[SEUIL] {label} : {a['seuils'] if a['seuils'] else 'aucun seuil (f ≤ 10 % puis ≥ 50 % entre deux doses voisines)'}")
        c = a["contre_temperature"]

        def fmt_eq(x):
            if isinstance(x, str):
                return x
            return f"T_eq {x['T_eq']:.2f}, G1 à T_eq {x['G1_T']:.3f}, bascules à T_eq {x['f_T']:.1%}, même {x['meme']}"
        print(f"[TEMP-ÉQ] {label} : {c['verdict']} — " + " ; ".join(f"p={p} : {fmt_eq(x)}" for p, x in c["par_dose"].items()))
        print(f"[THÈSE] {label} (doses faibles = maintien au bord de la bascule) : {a['these']}")
        for p in (DOSES[2], DOSES[4], DOSES[-1]):
            print(f"[EXTRAIT] {label} dose {p} prompt {SEL[0]} fin : {tok.decode(res[('dose', p)][2][0, GEN_LEN - 48:].tolist())!r}")
        print(f"[EXTRAIT] {label} T=1.1 prompt {SEL[0]} fin : {tok.decode(res[('T', 1.1)][2][0, GEN_LEN - 48:].tolist())!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a2.npz"), sel=np.asarray(SEL), doses=np.asarray(DOSES),
                            t_arms=np.asarray(T_ARMS),
                            **{f"{k}_dose_{p}": res[("dose", p)][i] for p in DOSES for i, k in enumerate(("S", "sur", "tok"))},
                            **{f"{k}_T_{T}": res[("T", T)][i] for T in T_ARMS for i, k in enumerate(("S", "sur", "tok"))},
                            **{f"{k}_add_{x}": res[("add", x)][i] for x in ADD_ARMS for i, k in enumerate(("S", "tok"))})
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min")
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}")
        ALL[label] = {"these": f"NON TESTABLE ({type(e).__name__})"}
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"\n{'=' * 100}")
for label, a in ALL.items():
    print(f"[SYNTHÈSE] {label} : thèse (doses faibles = maintien au bord de la bascule) {a['these']}")
with open(os.path.join(OUT, "resume_a2.json"), "w") as fh:
    json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "modeles": ALL}, fh,
              ensure_ascii=False, indent=1, default=str)
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

# %% [markdown]
# # SGZ A1c — Le retour après un choc vient-il de la trajectoire ? + chocs par corruption du texte
#
# **Suite de A1b (24 oct. 2026).** A1b : RETOUR à T = 0,6, INDÉTERMINÉ à 0,8, pas de sur-compensation, choc doux inerte,
# thèse INDÉTERMINÉE. Limite de fond : un retour monotone est aussi ce que donne l'**oubli passif** (le jeton bizarre perd
# son influence quand le contexte s'allonge), sans aucune rétroaction.
#
# **Idée du test.** On décompose l'effet du choc en deux parts (analyse de médiation) :
# - **bras libre** (comme A1b) : après le choc, le modèle génère sa propre suite ; R_libre(k) = S(choc) − S(témoin) ;
# - **bras rejoué** : après le choc, on force les jetons du **témoin** (teacher forcing, une passe avant, sans tirage) ;
#   R_rejoué(k) mesure l'effet direct du jeton choqué dans le contexte, trajectoire tenue fixe.
# - **Effet de la trajectoire** I(k) = R_libre(k) − R_rejoué(k) : ce que les jetons générés par le modèle ajoutent.
#   Une rétroaction négative par la trajectoire prédit I < 0 (le modèle compense) ; l'oubli passif prédit I ≈ 0 ;
#   une rétroaction positive prédit I > 0 (le modèle amplifie le choc).
#
# **Volet E (question d'Erick, 24 oct.) : chocs par corruption du texte, thème et vocabulaire technique gardés.** Au pas
# t₀, on corrompt les 32 derniers jetons générés par le modèle (son propre texte, sur un thème technique avancé), puis il
# continue : points au lieu des espaces ; lettres inversées ; un chiffre à la place d'une lettre ; mélange français et
# anglais ; fautes d'orthographe. Bras de référence : « identité » (même aller-retour décodage → encodage, sans
# corruption). On mesure l'effet sur l'entropie, le retour, la contagion du style et l'effet de la trajectoire.
#
# **Dispositif.** Qwen3-1.7B-Base. Volet F : 768 prompts (64 écrits à la main + 704 gabarits), T = 0,6 et 0,8, 160 jetons,
# choc fort (rang 1000–5000) au pas t₀ = 64. Volet E : 256 prompts à thème technique, T = 0,8. Hasard commun entre bras
# (même bruit de Gumbel par prompt et par pas). Nouvelle graine (réplication indépendante de A1b). Correction : les
# gabarits A1b avaient 32 phrases agrammaticales (« de le jazz ») ; ici « du », « des », « au », « aux ».
#
# ## Critères (fixés avant toute exécution ; bootstrap sur les prompts, B = 1000)
#
# - **F, effet de la trajectoire** (T = 0,6 et 0,8, choc fort) : D = moyenne de R_rejoué sur k = 2–31 (effet direct) ;
#   I = moyenne de R_libre − R_rejoué sur k = 2–31 (k = 1 est identique par construction) ; marge m = 0,3 × D.
#   NON TESTABLE si IC95(D) contient 0 ; ÉQUIVALENT (pas d'effet de trajectoire notable) si IC95(I) ⊂ [−m ; +m] ;
#   sinon COMPENSATION si IC95(I) < 0 ; AMPLIFICATION si IC95(I) > 0 ; sinon INDÉTERMINÉ.
# - **Verdict de la thèse, forme « rétroaction par la trajectoire »** : SOUTENUE si COMPENSATION à T = 0,6 et 0,8 ;
#   RÉFUTÉE si, aux deux T, ÉQUIVALENT ou AMPLIFICATION ; sinon INDÉTERMINÉE.
# - **E1, effet de la corruption** (k = 1–16, contre le bras identité) : HAUSSE si IC95 > 0 ; BAISSE si IC95 < 0 ;
#   sinon NON DÉTECTÉ.
# - **E2, contagion du style** (taux du détecteur propre à chaque corruption dans la suite, contre identité) :
#   CONTAGION si IC95 > 0 sur k = 1–32 ; PERSISTE si IC95 > 0 sur k = 64–95.
# - **E3, retour** (k = 64–95) : comme B1 avec marge 0,3 × |E1| ; NON TESTABLE si E1 = NON DÉTECTÉ.
# - **E4, effet de la trajectoire** : règle F, signe ramené à celui de l'effet direct (descriptif : puissance non calculée).
# - **B1 (réplication, descriptif)** : R_libre sur k = 64–95, marge 0,3 × E (k = 1–16).
# - NON INTERPRÉTABLE en SMOKE_TEST ; NON TESTABLE si le témoin positif échoue ; arrêt si une vérification d'alignement
#   échoue (écart d'entropie moyen > 0,1 entre une passe rejouée et la génération pour un même contexte).
#
# **Puissance (calculée d'avance sur A1b).** Écart-type par prompt de R_libre(k = 2–31) : 0,227 (T = 0,6), 0,221 (0,8).
# Avec 768 prompts, demi-IC de I ≈ 0,016 si Var(I) = Var(R_libre), ≈ 0,023 si elle est double. Si D ≈ 0,1 (m ≈ 0,03) :
# puissance d'équivalence (I vrai = 0) ≈ 0,91 ou 0,47 ; détection d'un |I| = m ≈ 0,95 ou 0,74. Avec 256 prompts
# (proposition initiale), l'équivalence était impossible (puissance ≈ 0).
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# F, à chaque T : AMPLIFICATION 40 %, ÉQUIVALENT 25 %, COMPENSATION 15 %, INDÉTERMINÉ 20 % (dans A1b, après un jeton
# bizarre le modèle enchaîne souvent sur d'autres jetons bizarres, comme une liste de noms propres). Thèse : SOUTENUE 12 %,
# RÉFUTÉE 50 %, INDÉTERMINÉE 38 %. E1 = HAUSSE : chiffres, inversion, fautes 75 % chacun ; mélange 60 % ; points 40 %
# (BAISSE 40 %). E2 CONTAGION : points 85 %, chiffres 70 %, mélange 60 %, inversion et fautes 50 %.
#
# ## Limites connues
#
# - Le bras rejoué n'est pas un oubli « pur » : la suite du témoin peut mal s'accorder au jeton choqué. I mesure ce
#   qu'ajoute la suite propre du modèle, comparée à la suite qu'il aurait écrite sans choc.
# - Une correction interne (têtes d'attention qui ignorent le jeton anormal) reste dans l'effet direct : ce test ne vise
#   que la rétroaction qui passe par les jetons générés.
# - Détecteurs de style simples (regex, lexique tiré des textes témoins) ; un seul modèle.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : 30 à 45 min [estimation]. Budget : 3 h.
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[PROGRESSION]`, `[F]`, `[THÈSE]`, `[E1]` à `[E4]`.

# %%
import gc
import difflib
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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1c" if os.path.isdir("/kaggle/working") else "sgz_a1c_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6, 0.8]
T_CORR = 0.8
Q = 1.45
N_PROMPTS, N_CORR, GEN_LEN, T0, SPAN = 768, 256, 160, 64, 32
RANK_LO, RANK_HI = 1000, 5000
EARLY, MID_F, CONT_EARLY, LATE = (1, 16), (2, 31), (1, 32), (64, 95)
MARGE_REL, P_MOT = 0.3, 0.5
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
T_RUN0 = time.time()
CORR_TYPES = ["points", "inversion", "chiffres", "melange", "fautes"]
CORR_ARMS = ["identite"] + CORR_TYPES
MODELS = [("Qwen3-1.7B-Base", ["Qwen/Qwen3-1.7B-Base"])]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, N_CORR, GEN_LEN, T0, SPAN = 16, 8, 72, 24, 8
    RANK_LO, RANK_HI = 20, 80
    EARLY, MID_F, CONT_EARLY, LATE = (1, 4), (2, 8), (1, 8), (33, 47)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]
K = GEN_LEN - T0 - 1                       # k = 1..K après le choc
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} (volet F), T={T_CORR} (volet E) ; prompts F={N_PROMPTS}, E={N_CORR} ; longueur={GEN_LEN} t0={T0} "
      f"segment corrompu={SPAN} jetons ; modèles={[m[0] for m in MODELS]}")
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
    CORR_IDX = list(range(N_CORR))
else:                                      # volet E : gabarits techniques, 16 premiers thèmes × 8 débuts × 2 langues
    CORR_IDX = [i for i, (_, _, tech, ti) in enumerate(ALL_P) if tech and ti < 16][:N_CORR]
assert len(CORR_IDX) == N_CORR
print(f"[PROMPTS] {len(PROMPTS)} prompts ({min(64, len(PROMPTS))} écrits à la main, {max(0, len(PROMPTS) - 64)} gabarits) ; "
      f"volet E : {len(CORR_IDX)} prompts techniques ({sum(LANG[i] == 'fr' for i in CORR_IDX)} en français)")

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


def win(R, w):
    return R[:, w[0] - 1:w[1]].mean(1)


def trajectoire(Rlib, Rrej, boots, interpretable=True):
    """Volet F / E4 : effet direct D (bras rejoué) et effet de la trajectoire I = libre − rejoué, signe ramené à D."""
    D = win(Rrej, MID_F)
    cD = ci([D[ix].mean() for ix in boots])
    sg = 1.0 if D.mean() >= 0 else -1.0
    I = sg * (win(Rlib, MID_F) - D)
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


def effet(R, boots, interpretable=True):
    E = win(R, EARLY)
    c = ci([E[ix].mean() for ix in boots])
    v = "NON INTERPRÉTABLE" if not interpretable else ("HAUSSE" if c[0] > 0 else "BAISSE" if c[1] < 0 else "NON DÉTECTÉ")
    return {"E": float(E.mean()), "ci": c, "verdict": v}


def retour(R, boots, interpretable=True):
    E, late = win(R, EARLY), win(R, LATE)
    ce, cl = ci([E[ix].mean() for ix in boots]), ci([late[ix].mean() for ix in boots])
    marge = MARGE_REL * abs(E.mean())
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif ce[0] <= 0 <= ce[1]:
        v = "NON TESTABLE (choc inerte)"
    elif cl[0] >= -marge and cl[1] <= marge:
        v = "RETOUR"
    elif cl[0] > marge or cl[1] < -marge:
        v = "PAS DE RETOUR"
    else:
        v = "INDÉTERMINÉ"
    return {"E": float(E.mean()), "ci_E": ce, "R_tard": float(late.mean()), "ci_tard": cl, "marge": float(marge), "verdict": v}


def contagion(d, boots, interpretable=True, mot="CONTAGION"):
    c = ci([d[ix].mean() for ix in boots])
    v = "NON INTERPRÉTABLE" if not interpretable else (mot if c[0] > 0 else "NON DÉTECTÉE")
    return {"diff": float(np.mean(d)), "ci": c, "verdict": v}


# Témoin positif : réponses synthétiques connues (bruit par prompt réaliste, écart-type ≈ 0,2 comme A1b)
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
kk = np.arange(1, K + 1)


def synth_pair(factor, amp=1.0, decay=0.8, Pn=Pp):
    direct = amp * decay ** kk
    rej = direct[None, :] + rng_pc.normal(0, 0.05, (Pn, 1)) + rng_pc.normal(0, 0.1, (Pn, K))
    lib = factor * direct[None, :] + rng_pc.normal(0, 0.2, (Pn, 1)) + rng_pc.normal(0, 0.3, (Pn, K))
    return lib, rej


bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
bt_small = [rng_pc.integers(0, 256, 256) for _ in range(200)]
noise256 = lambda mu: mu * 0.5 ** np.minimum(kk, 40)[None, :] + rng_pc.normal(0, 0.3, (256, K))
checks = [
    ("F compensation", trajectoire(*synth_pair(0.4), bt)["verdict"], "COMPENSATION"),
    ("F équivalent", trajectoire(*synth_pair(1.0), bt)["verdict"], "ÉQUIVALENT"),
    ("F amplification", trajectoire(*synth_pair(1.6), bt)["verdict"], "AMPLIFICATION"),
    ("F effet direct nul", trajectoire(*synth_pair(1.0, amp=0.0), bt)["verdict"], "NON TESTABLE (effet direct absent)"),
    ("E1 hausse", effet(noise256(1.0), bt_small)["verdict"], "HAUSSE"),
    ("E1 baisse", effet(noise256(-1.0), bt_small)["verdict"], "BAISSE"),
    ("E1 rien", effet(noise256(0.0), bt_small)["verdict"], "NON DÉTECTÉ"),
    ("E3 retour", retour(noise256(1.0), bt_small)["verdict"], "RETOUR"),
]
SAMPLE = {"fr": "La chromodynamique quantique décrit l'interaction forte entre les quarks et les gluons, dont la liberté "
                "asymptotique explique le comportement des hadrons à très haute énergie.",
          "en": "Quantum chromodynamics describes the strong interaction between quarks and gluons, whose asymptotic "
                "freedom explains the behaviour of hadrons at very high energy."}
for lang, s in SAMPLE.items():
    lex = {w.lower() for w in RE_WORD.findall(s)}
    for j, kind in enumerate(CORR_TYPES):
        c = corrupt(s, kind, lang, np.random.default_rng([SEED, j]))
        ok = c != s and taux_style(c, kind, lang, lex) > taux_style(s, kind, lang, lex)
        checks.append((f"texte {lang} {kind}", "DÉTECTÉ" if ok else "RATÉ", "DÉTECTÉ"))
for name, got, exp in checks:
    print(f"[TÉMOIN POSITIF] {name:<20} : {got:<36} (attendu {exp})")
for lang, s in SAMPLE.items():
    for j, kind in enumerate(CORR_TYPES):
        print(f"[EXEMPLE] {lang} {kind:<9} : {corrupt(s, kind, lang, np.random.default_rng([SEED, j]))[:110]}")
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
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
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
# ---------- Génération : bras empilés (témoin, choc fort), bras ramifiés (corruptions), bras rejoués ----------
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
        self.budget = 48 if small else 16
        self.t_first, self.lots_faits = None, 0

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
        """Génère nsteps jetons après chaque contexte, bruit de Gumbel graine (prompt, step0 + s).
        force(s, z, nxt) peut remplacer des jetons (appariement avant t₀, choc à t₀). Entropie de p₁ (T = 1)."""
        n = len(ctxs)
        S, sur, tk = (np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int64))
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
            sur[:, s] = (-torch.log_softmax(z.double(), -1).gather(1, nxt[:, None])[:, 0]).cpu().numpy()
            S[:, s], tk[:, s] = e1, nxt.cpu().numpy()
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = pos[:, -1:] + 1
        del past, out
        return S, sur, tk

    @torch.no_grad()
    def replay(self, ctxs, conts, chunk=192):
        """Une passe avant sur contexte + suite forcée ; entropie de p₁ aux len(cont) + 1 pas qui suivent le contexte."""
        base, head = self.model.model, self.model.get_output_embeddings()
        n, Kp = len(ctxs), len(conts[0]) + 1
        S = np.zeros((n, Kp), np.float32)
        ids, attn, L = self.left_pad([c + s for c, s in zip(ctxs, conts)])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
        starts = [L - len(s) - 1 for s in conts]                    # dernier jeton du contexte de chaque ligne
        idx = torch.tensor([[st + i for i in range(Kp)] for st in starts], device=h.device)
        hk = h[torch.arange(n, device=h.device)[:, None], idx].reshape(n * Kp, -1)
        flat = np.zeros(n * Kp, np.float64)
        for c0 in range(0, n * Kp, chunk):
            z = head(hk[c0:c0 + chunk]).float()                  # pas de masque : même calcul que pendant la génération
            flat[c0:c0 + chunk] = entropies_from_logits(z, [Q])[Q].cpu().numpy()
        S[:] = flat.reshape(n, Kp)
        return S

    def progress(self, t_lot, n_lots_total):
        self.lots_faits += 1
        if self.t_first is None:
            self.t_first = t_lot
            est = t_lot * n_lots_total / 60
            print(f"[PROGRESSION] {self.label} : premier lot en {t_lot:.1f} s ; estimation ~{est:.0f} min (génération)", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")


@torch.no_grad()
def run_main(g, enc, n_lots_total):
    """Volet F : témoin et choc fort empilés (appariement exact avant t₀), comme A1b."""
    P = len(enc)
    forced_rank = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, P)
    res = {}
    npb = max(1, g.budget // 2)
    for T in T_MAIN:
        t0 = time.time()
        for a in ("temoin", "fort"):
            for key in ("S", "sur", "tok"):
                res[(key, T, a)] = np.zeros((P, GEN_LEN), np.int64 if key == "tok" else np.float32)
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
                return nxt

            S, sur, tk = g.sample([enc[r] for r in rows_p] * 2, rows_p * 2, T, GEN_LEN, 0, force)
            for ai, a in enumerate(("temoin", "fort")):
                sl = slice(ai * n, (ai + 1) * n)
                res[("S", T, a)][rows_p], res[("sur", T, a)][rows_p], res[("tok", T, a)][rows_p] = S[sl], sur[sl], tk[sl]
            g.progress(time.time() - tl, n_lots_total)
        ref, fo = res[("tok", T, "temoin")], res[("tok", T, "fort")]
        same = np.array_equal(ref[:, :T0], fo[:, :T0])
        dS = float(np.max(np.abs(res[("S", T, "temoin")][:, :T0 + 1] - res[("S", T, "fort")][:, :T0 + 1])))
        chg = float(np.mean(ref[:, T0] != fo[:, T0]))
        print(f"[VÉRIF] {g.label} T={T} choc fort : identique avant t0 : {same} ; écart d'entropie max jusqu'à t0 : {dS:.1e} ; "
              f"jeton changé à t0 : {chg:.0%} ; rang médian {int(np.median(rang))}", flush=True)
        if not same or dS > 1e-2 or chg < 0.99:
            raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
        print(f"[MESURE] {g.label} T={T} : {time.time() - t0:.0f} s", flush=True)
    return res


@torch.no_grad()
def run_replay_main(g, enc, res):
    """Bras rejoués du volet F : témoin rejoué (contrôle d'alignement) et choc suivi de la suite du témoin."""
    P = len(enc)
    for T in T_MAIN:
        ref, fo = res[("tok", T, "temoin")], res[("tok", T, "fort")]
        for a in ("temoin", "fort"):
            res[("Srej", T, a)] = np.zeros((P, GEN_LEN), np.float32)
        nb = max(1, g.budget // 3)
        for b0 in range(0, P, nb):
            rows = list(range(b0, min(P, b0 + nb)))
            conts_t = [ref[r, :GEN_LEN - 1].tolist() for r in rows]
            conts_f = [c[:T0] + [int(fo[r, T0])] + c[T0 + 1:] for c, r in zip(conts_t, rows)]
            S = g.replay([enc[r] for r in rows] * 2, conts_t + conts_f)
            res[("Srej", T, "temoin")][rows], res[("Srej", T, "fort")][rows] = S[:len(rows)], S[len(rows):]
        d_all = np.abs(res[("Srej", T, "temoin")] - res[("S", T, "temoin")])
        d1 = np.abs(res[("Srej", T, "fort")][:, T0 + 1] - res[("S", T, "fort")][:, T0 + 1])
        print(f"[VÉRIF] {g.label} T={T} rejoué contre généré (même contexte) : témoin écart moyen {d_all.mean():.1e}, "
              f"max {d_all.max():.1e} ; choc à k=1 écart moyen {d1.mean():.1e}", flush=True)
        if d_all.mean() > 0.1 or d1.mean() > 0.1:
            raise RuntimeError("bras rejoué mal aligné sur la génération : résultats non interprétables")
    return res


@torch.no_grad()
def run_corr(g, enc, res, n_lots_total):
    """Volet E : on corrompt les SPAN derniers jetons du témoin (T_CORR) avant t₀, puis génération libre et rejouée."""
    ref = res[("tok", T_CORR, "temoin")]
    ctx, txt = {a: [] for a in CORR_ARMS}, {a: [] for a in CORR_ARMS}
    exact = []
    for r in CORR_IDX:
        span = ref[r, T0 - SPAN:T0].tolist()
        s_txt = g.tok.decode(span)
        exact.append(g.tok.encode(s_txt, add_special_tokens=False) == span)
        for j, a in enumerate(CORR_ARMS):
            c = corrupt(s_txt, a, LANG[r], np.random.default_rng([SEED, r, j]))
            txt[a].append(c)
            ctx[a].append(enc[r] + ref[r, :T0 - SPAN].tolist() + g.tok.encode(c, add_special_tokens=False))
    na, n_c = len(CORR_ARMS), len(CORR_IDX)
    out = {("S", a): np.zeros((n_c, K), np.float32) for a in CORR_ARMS}
    out.update({("tok", a): np.zeros((n_c, K), np.int64) for a in CORR_ARMS})
    out.update({("Srej", a): np.zeros((n_c, K), np.float32) for a in CORR_ARMS})
    npb = max(1, g.budget // na)
    t0 = time.time()
    for b0 in range(0, n_c, npb):
        tl = time.time()
        loc = list(range(b0, min(n_c, b0 + npb)))
        ctxs = [ctx[a][i] for a in CORR_ARMS for i in loc]           # bras empilés par lot de prompts
        S, _, tk = g.sample(ctxs, [CORR_IDX[i] for a in CORR_ARMS for i in loc], T_CORR, K, T0)
        for ai, a in enumerate(CORR_ARMS):
            out[("S", a)][loc], out[("tok", a)][loc] = S[ai * len(loc):(ai + 1) * len(loc)], tk[ai * len(loc):(ai + 1) * len(loc)]
        conts = [ref[CORR_IDX[i], T0:T0 + K - 1].tolist() for a in CORR_ARMS for i in loc]
        Sr = np.concatenate([g.replay(ctxs[c0:c0 + 16], conts[c0:c0 + 16]) for c0 in range(0, len(ctxs), 16)])
        for ai, a in enumerate(CORR_ARMS):
            out[("Srej", a)][loc] = Sr[ai * len(loc):(ai + 1) * len(loc)]
        g.progress(time.time() - tl, n_lots_total)
    ex = np.array(exact)
    s0 = res[("S", T_CORR, "temoin")][CORR_IDX, T0]
    d0 = float(np.abs(out[("S", "identite")][:, 0] - s0)[ex].mean()) if ex.any() else float("nan")
    agree = float((out[("tok", "identite")] == ref[CORR_IDX, T0:T0 + K])[ex].mean()) if ex.any() else float("nan")
    print(f"[VÉRIF] {g.label} volet E : aller-retour décodage → encodage exact pour {ex.mean():.0%} des segments ; "
          f"identité contre témoin au premier pas (segments exacts) : écart d'entropie moyen {d0:.1e} ; "
          f"jetons identiques sur la suite : {agree:.1%}", flush=True)
    if ex.any() and d0 > 0.1:
        raise RuntimeError("bras identité mal aligné sur le témoin : résultats non interprétables")
    for a in CORR_TYPES:
        ch = np.mean([change_rate(x, y) for x, y in zip(txt["identite"], txt[a])])
        dl = np.mean([len(c) - len(i) for c, i in zip(ctx[a], ctx["identite"])])
        same = np.mean([x == y for x, y in zip(txt["identite"], txt[a])])
        print(f"[VÉRIF] {g.label} corruption {a:<9} : caractères changés {ch:.1%} ; jetons en plus dans le segment {dl:+.1f} ; "
              f"segments inchangés {same:.0%}", flush=True)
    print(f"[MESURE] {g.label} volet E : {time.time() - t0:.0f} s", flush=True)
    out["txt"], out["exact"] = txt, ex
    return out

# %%
# ---------- Analyse ----------
TRANCHES = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 95)]


def tranches(R):
    return [float(R[:, x - 1:y].mean()) for x, y in TRANCHES if y <= R.shape[1]]


def analyse(res, cor, tok, interpretable):
    ok = interpretable and POSITIVE_CONTROL_OK
    rng = np.random.default_rng(SEED + 3)
    P = res[("S", T_MAIN[0], "temoin")].shape[0]
    boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]
    out = {"F": {}, "B1": {}, "courbes": {}, "E": {}}
    for T in T_MAIN:
        Rlib = (res[("S", T, "fort")] - res[("S", T, "temoin")])[:, T0 + 1:].astype(float)
        Rrej = (res[("Srej", T, "fort")] - res[("Srej", T, "temoin")])[:, T0 + 1:].astype(float)
        out["F"][T] = trajectoire(Rlib, Rrej, boots, ok)
        out["B1"][T] = retour(Rlib, boots, ok)
        out["courbes"][f"{T}|libre"], out["courbes"][f"{T}|rejoué"] = tranches(Rlib), tranches(Rrej)
        out["courbes"][f"{T}|trajectoire"] = tranches(Rlib - Rrej)
    v = [out["F"][T]["verdict"] for T in T_MAIN]
    if not ok:
        these = "NON INTERPRÉTABLE" if not interpretable else "NON TESTABLE (témoin positif raté)"
    elif all(x == "COMPENSATION" for x in v):
        these = "SOUTENUE"
    elif all(x in ("ÉQUIVALENT", "AMPLIFICATION") for x in v):
        these = "RÉFUTÉE"
    else:
        these = "INDÉTERMINÉE"
    out["these"] = these
    # Volet E
    nC = len(CORR_IDX)
    bootsC = [rng.integers(0, nC, nC) for _ in range(BOOT_N)]
    lex = set()
    for T in T_MAIN:
        for r in range(P):
            lex |= {w.lower() for w in RE_WORD.findall(tok.decode(res[("tok", T, "temoin")][r].tolist()))}
    for p in PROMPTS:
        lex |= {w.lower() for w in RE_WORD.findall(p)}

    def suite_txt(a, i, w):
        return tok.decode(cor[("tok", a)][i, w[0] - 1:w[1]].tolist())

    for a in CORR_TYPES:
        R = (cor[("S", a)] - cor[("S", "identite")]).astype(float)
        Rr = (cor[("Srej", a)] - cor[("Srej", "identite")]).astype(float)
        e1 = effet(R, bootsC, ok)
        e3 = retour(R, bootsC, ok) if e1["verdict"] in ("HAUSSE", "BAISSE") else {"verdict": "NON TESTABLE (E1 non détecté)"}
        rates = {}
        for wname, w in (("tot", CONT_EARLY), ("tard", LATE)):
            for arm in (a, "identite"):
                rates[(wname, arm)] = np.array([taux_style(suite_txt(arm, i, w), a, LANG[CORR_IDX[i]], lex) for i in range(nC)])
        e2a = contagion(rates[("tot", a)] - rates[("tot", "identite")], bootsC, ok, "CONTAGION")
        e2b = contagion(rates[("tard", a)] - rates[("tard", "identite")], bootsC, ok, "PERSISTE")
        e2a["taux"] = [float(rates[("tot", a)].mean()), float(rates[("tot", "identite")].mean())]
        e2b["taux"] = [float(rates[("tard", a)].mean()), float(rates[("tard", "identite")].mean())]
        e4 = trajectoire(R, Rr, bootsC, ok)
        out["E"][a] = {"E1": e1, "E2_tot": e2a, "E2_tard": e2b, "E3": e3, "E4": e4, "courbe": tranches(R),
                       "courbe_rejoue": tranches(Rr)}
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
        enc = [tok.encode(p)[:12] if SMOKE_TEST else tok.encode(p) for p in PROMPTS]
        n_main = len(T_MAIN) * math.ceil(len(enc) / max(1, g.budget // 2))
        n_corr = math.ceil(len(CORR_IDX) / max(1, g.budget // len(CORR_ARMS))) * K / GEN_LEN
        res = run_main(g, enc, n_main + n_corr)
        res = run_replay_main(g, enc, res)
        cor = run_corr(g, enc, res, n_main + n_corr)
        a = analyse(res, cor, tok, interpretable)
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label}")
        for T in T_MAIN:
            r = a["F"][T]
            print(f"[F] {label} T={T} : {r['verdict']:<14} effet direct D(k={MID_F[0]}–{MID_F[1]}) = {r['D']:+.4f} "
                  f"IC95 [{r['ci_D'][0]:+.4f} ; {r['ci_D'][1]:+.4f}] ; trajectoire I = {r['I']:+.4f} "
                  f"IC95 [{r['ci_I'][0]:+.4f} ; {r['ci_I'][1]:+.4f}] ; marge ±{r['marge']:.4f} ; I/D = {r['rapport']:+.2f}")
        for T in T_MAIN:
            r = a["B1"][T]
            print(f"[B1 desc] {label} T={T} : {r['verdict']} — E = {r['E']:+.4f} ; R(k={LATE[0]}–{LATE[1]}) = {r['R_tard']:+.4f} "
                  f"IC95 [{r['ci_tard'][0]:+.4f} ; {r['ci_tard'][1]:+.4f}] ; marge ±{r['marge']:.4f}")
        for key, c in a["courbes"].items():
            print(f"[COURBE] {label} {key:<15} : R par tranches de k (1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–95) : "
                  + " ".join(f"{v:+.3f}" for v in c))
        print(f"[THÈSE] {label} (rétroaction par la trajectoire) : {a['these']}")
        for kind in CORR_TYPES:
            e = a["E"][kind]
            print(f"[E1] {label} {kind:<9} : {e['E1']['verdict']:<11} R(k={EARLY[0]}–{EARLY[1]}) = {e['E1']['E']:+.4f} "
                  f"IC95 [{e['E1']['ci'][0]:+.4f} ; {e['E1']['ci'][1]:+.4f}]")
            for key, nm in (("E2_tot", f"k={CONT_EARLY[0]}–{CONT_EARLY[1]}"), ("E2_tard", f"k={LATE[0]}–{LATE[1]}")):
                r = e[key]
                print(f"[E2] {label} {kind:<9} {nm:<8} : {r['verdict']:<12} taux corrompu {r['taux'][0]:.4f} contre identité "
                      f"{r['taux'][1]:.4f} ; écart IC95 [{r['ci'][0]:+.4f} ; {r['ci'][1]:+.4f}]")
            r = e["E3"]
            extra = (f" — R(k={LATE[0]}–{LATE[1]}) = {r['R_tard']:+.4f} IC95 [{r['ci_tard'][0]:+.4f} ; {r['ci_tard'][1]:+.4f}] ; "
                     f"marge ±{r['marge']:.4f}") if "R_tard" in r else ""
            print(f"[E3] {label} {kind:<9} : {r['verdict']}{extra}")
            r = e["E4"]
            print(f"[E4] {label} {kind:<9} : {r['verdict']} — D = {r['D']:+.4f} IC95 [{r['ci_D'][0]:+.4f} ; {r['ci_D'][1]:+.4f}] ; "
                  f"I = {r['I']:+.4f} IC95 [{r['ci_I'][0]:+.4f} ; {r['ci_I'][1]:+.4f}] ; marge ±{r['marge']:.4f}")
            print(f"[COURBE] {label} corruption {kind:<9} : " + " ".join(f"{v:+.3f}" for v in e["courbe"]))
        for p in (0, 64 if len(PROMPTS) > 64 else 0):
            a0 = max(0, T0 - 12)
            for arm in ("temoin", "fort"):
                print(f"[EXTRAIT] {label} T=0.8 prompt {p} {arm:<6} : {tok.decode(res[('tok', 0.8, arm)][p, a0:T0 + 40].tolist())!r}")
        for i in (0, len(CORR_IDX) // 2 + 1):
            for arm in CORR_ARMS:
                print(f"[EXTRAIT] {label} volet E prompt {CORR_IDX[i]} {arm:<9} : segment {cor['txt'][arm][i][-80:]!r} → suite "
                      f"{tok.decode(cor[('tok', arm)][i, :40].tolist())!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a1c.npz"), T_main=np.asarray(T_MAIN), corr_idx=np.asarray(CORR_IDX),
                            **{f"{k}_{T}_{arm}": res[(k, T, arm)] for k in ("S", "sur", "tok", "Srej") for T in T_MAIN
                               for arm in ("temoin", "fort")},
                            **{f"corr_{k}_{arm}": cor[(k, arm)] for k in ("S", "tok", "Srej") for arm in CORR_ARMS},
                            corr_exact=cor["exact"])
        with open(os.path.join(OUT, f"{label}_segments_a1c.json"), "w") as fh:
            json.dump({"corr_idx": CORR_IDX, "segments": cor["txt"]}, fh, ensure_ascii=False)
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
    print(f"[SYNTHÈSE] {label} : thèse (rétroaction par la trajectoire) {a['these']}")
with open(os.path.join(OUT, "resume_a1c.json"), "w") as fh:
    json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "modeles": ALL}, fh,
              ensure_ascii=False, indent=1, default=str)
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

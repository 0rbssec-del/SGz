# %% [markdown]
# # SGZ B1b — Après une erreur factuelle, l'explication libre devient-elle plus incertaine, au-delà de la surprise ? (branche 1)
#
# **Pourquoi B1b.** B1 (chaînes de calcul) n'a rien pu trancher : la suite d'une chaîne est presque déterministe (entropie
# médiane 0,0004 chez Qwen3), donc ni l'erreur ni sa surprise ne pouvaient la faire bouger. Fait net, en revanche : après son
# erreur, le modèle recopie la mauvaise valeur dans 100 % des suites, sans hésiter. B1b reprend le même plan, mais l'erreur est
# suivie d'une **explication libre**, où l'entropie a de la place, et ajoute deux garde-fous fixés d'avance : un contrôle de
# plancher et une marge d'équivalence absolue.
#
# **Dispositif.** Questions vérifiables automatiquement, en français et en anglais, avec trois exemples résolus :
# `Élément : tungstène` / `Numéro atomique : 74. Le tungstène est…` ; `Élément : …` / `Symbole chimique : W. …` ;
# `Pays : Australie` / `Capitale : Canberra. …`. 118 éléments (numéro atomique et symbole) et 106 pays aux capitales sans
# ambiguïté, soit 676 questions (le fer, le sodium et le Japon servent d'exemples et sont retirés).
# 1. **Sélection** : 8 réponses libres par question (T = 0,6). On garde les questions où le modèle a donné au moins une fois
#    la bonne réponse et au moins une fois une mauvaise ; C = sa bonne réponse la plus fréquente, W = sa mauvaise la plus
#    fréquente (deux choix que le modèle a réellement faits).
# 2. **Surprise** : s(v) = −log p(« v. ») dans la loi du modèle (T = 1) ; Δs = s(W) − s(C).
# 3. **Branches** (explication libre de 40 jetons au plus, même bruit de Gumbel à chaque position k) : **C**, **W**, **nul**
#    (C avec un autre bruit), **reprise** (C régénérée dans d'autres lots).
#
# ## Mesures
#
# H = entropie de Tsallis (q = 1,45) moyenne des lois des jetons k = 1 … 30 de l'explication, coupée au premier saut de ligne.
# Par question : T = H(W) − H(C) ; T(nul) = H(nul) − H(C).
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les questions, B = 1000)
#
# - **Plancher** : l'entropie médiane après la bonne réponse doit être ≥ 0,02 ; sinon E1 et E2 NON TESTABLES (plancher).
# - **Témoin négatif** : T(nul) sans biais (IC95 contenant 0) ; sinon NON INTERPRÉTABLE.
# - **E1 (principal), réaction à l'erreur à surprise égale** : T = a + b·Δs. Marge m = 0,10 × H(C) moyen (10 % de l'entropie
#   de base). ABSENTE si l'IC95 de a ⊂ [−m ; +m] ; sinon RÉACTION À L'ERREUR si la borne basse > 0 ; ENGAGEMENT si la borne
#   haute < 0 ; sinon INDÉTERMINÉ (même ordre que la règle F). NON TESTABLE si moins de 10 % des questions de chaque côté de
#   Δs = 0 ou moins de 80 questions.
# - **E2, quand l'erreur est le choix le plus probable** (Δs < 0) : moyenne de T. L'ERREUR L'EMPORTE si la borne basse > 0 ;
#   LA SURPRISE L'EMPORTE si la borne haute < 0 ; sinon INDÉTERMINÉ ; NON TESTABLE si moins de 30 questions.
# - **Synthèse, thèse « le modèle réagit à sa propre erreur au-delà de la surprise »** : SOUTENUE si E1 = RÉACTION et E2 ≠ LA
#   SURPRISE L'EMPORTE pour les deux modèles ; RÉFUTÉE si E1 ∈ {ABSENTE, ENGAGEMENT} pour les deux ; DÉPEND DU MODÈLE si un
#   modèle suit le premier schéma et l'autre le second ; sinon INDÉTERMINÉE.
# - **Descriptifs** : T par domaine ; part des explications après W qui citent la bonne réponse (correction) ; pour les
#   numéros et symboles, part des explications qui parlent de l'élément de la mauvaise réponse (propagation du sens) ou de
#   l'élément demandé ; sous-ensemble apparié |Δs| ≤ 0,5 ; part des questions où T > 0 ; exemples d'explications.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Plancher franchi : 85 %. E1 : RÉACTION 30 %, ABSENTE 30 %, ENGAGEMENT 10 %, autre 30 %. Synthèse SOUTENUE 20 %, RÉFUTÉE
# 30 %. Après une mauvaise réponse sur un numéro atomique, l'explication parle de l'élément de ce mauvais numéro dans au moins
# 30 % des cas : 50 %.
#
# ## Limites connues
#
# - Faits encyclopédiques simples, format très court ; modèles de base en quelques exemples.
# - La correspondance des capitales passe par une liste d'alias écrite à la main (accents, tirets, variantes) : une variante
#   absente compterait comme une erreur.
# - Les erreurs sur des éléments lourds ou des pays peu connus peuvent refléter une ignorance plutôt qu'un savoir contredit.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~10 min pour Qwen3,
# ~15 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[SÉLECTION]`, `[VÉRIF]`, `[PLANCHER]`,
# `[TÉMOIN NÉGATIF]`, `[E1]`, `[E2]`, `[SYNTHÈSE]`.

# %%
import gc
import json
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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_b1b" if os.path.isdir("/kaggle/working") else "sgz_b1b_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = 0.6
Q = 1.45
K_ECH, N_MIN, N_SOUS = 8, 80, 30
LONG_REP, LONG_SUITE, KMAX = 10, 40, 30          # jetons de la réponse libre, de l'explication, fenêtre k = 1 … 30
PLANCHER = 0.02                                  # entropie médiane minimale après la bonne réponse
MARGE_REL = 0.10                                 # ABSENTE : |a| < 0,10 × H(C) moyen
ECART_APP = 0.5
COUV = 0.10
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261030
SEL = SEED + 31
FRAIS = SEED + 7919
ETAPE_SUITE = 1000
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    K_ECH, N_MIN, N_SOUS = 6, 10, 5
    LONG_REP, LONG_SUITE, KMAX = 8, 24, 20
    PLANCHER = 0.0
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
TRANCHES = [(lo, min(hi, KMAX)) for lo, hi in ((1, 5), (6, 15), (16, 30)) if lo <= KMAX]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} tirages={K_ECH} ; explication {LONG_SUITE} jetons, fenêtre k = 1–{KMAX} ; modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# %%
# ---------- Banque de questions ----------
# Éléments dans l'ordre des numéros atomiques 1 … 118 : (symbole, nom anglais, nom français)
ELEMENTS = [
    ("H", "hydrogen", "hydrogène"), ("He", "helium", "hélium"), ("Li", "lithium", "lithium"), ("Be", "beryllium", "béryllium"),
    ("B", "boron", "bore"), ("C", "carbon", "carbone"), ("N", "nitrogen", "azote"), ("O", "oxygen", "oxygène"),
    ("F", "fluorine", "fluor"), ("Ne", "neon", "néon"), ("Na", "sodium", "sodium"), ("Mg", "magnesium", "magnésium"),
    ("Al", "aluminium", "aluminium"), ("Si", "silicon", "silicium"), ("P", "phosphorus", "phosphore"), ("S", "sulfur", "soufre"),
    ("Cl", "chlorine", "chlore"), ("Ar", "argon", "argon"), ("K", "potassium", "potassium"), ("Ca", "calcium", "calcium"),
    ("Sc", "scandium", "scandium"), ("Ti", "titanium", "titane"), ("V", "vanadium", "vanadium"), ("Cr", "chromium", "chrome"),
    ("Mn", "manganese", "manganèse"), ("Fe", "iron", "fer"), ("Co", "cobalt", "cobalt"), ("Ni", "nickel", "nickel"),
    ("Cu", "copper", "cuivre"), ("Zn", "zinc", "zinc"), ("Ga", "gallium", "gallium"), ("Ge", "germanium", "germanium"),
    ("As", "arsenic", "arsenic"), ("Se", "selenium", "sélénium"), ("Br", "bromine", "brome"), ("Kr", "krypton", "krypton"),
    ("Rb", "rubidium", "rubidium"), ("Sr", "strontium", "strontium"), ("Y", "yttrium", "yttrium"), ("Zr", "zirconium", "zirconium"),
    ("Nb", "niobium", "niobium"), ("Mo", "molybdenum", "molybdène"), ("Tc", "technetium", "technétium"),
    ("Ru", "ruthenium", "ruthénium"), ("Rh", "rhodium", "rhodium"), ("Pd", "palladium", "palladium"), ("Ag", "silver", "argent"),
    ("Cd", "cadmium", "cadmium"), ("In", "indium", "indium"), ("Sn", "tin", "étain"), ("Sb", "antimony", "antimoine"),
    ("Te", "tellurium", "tellure"), ("I", "iodine", "iode"), ("Xe", "xenon", "xénon"), ("Cs", "caesium", "césium"),
    ("Ba", "barium", "baryum"), ("La", "lanthanum", "lanthane"), ("Ce", "cerium", "cérium"), ("Pr", "praseodymium", "praséodyme"),
    ("Nd", "neodymium", "néodyme"), ("Pm", "promethium", "prométhium"), ("Sm", "samarium", "samarium"),
    ("Eu", "europium", "europium"), ("Gd", "gadolinium", "gadolinium"), ("Tb", "terbium", "terbium"),
    ("Dy", "dysprosium", "dysprosium"), ("Ho", "holmium", "holmium"), ("Er", "erbium", "erbium"), ("Tm", "thulium", "thulium"),
    ("Yb", "ytterbium", "ytterbium"), ("Lu", "lutetium", "lutécium"), ("Hf", "hafnium", "hafnium"), ("Ta", "tantalum", "tantale"),
    ("W", "tungsten", "tungstène"), ("Re", "rhenium", "rhénium"), ("Os", "osmium", "osmium"), ("Ir", "iridium", "iridium"),
    ("Pt", "platinum", "platine"), ("Au", "gold", "or"), ("Hg", "mercury", "mercure"), ("Tl", "thallium", "thallium"),
    ("Pb", "lead", "plomb"), ("Bi", "bismuth", "bismuth"), ("Po", "polonium", "polonium"), ("At", "astatine", "astate"),
    ("Rn", "radon", "radon"), ("Fr", "francium", "francium"), ("Ra", "radium", "radium"), ("Ac", "actinium", "actinium"),
    ("Th", "thorium", "thorium"), ("Pa", "protactinium", "protactinium"), ("U", "uranium", "uranium"),
    ("Np", "neptunium", "neptunium"), ("Pu", "plutonium", "plutonium"), ("Am", "americium", "américium"),
    ("Cm", "curium", "curium"), ("Bk", "berkelium", "berkélium"), ("Cf", "californium", "californium"),
    ("Es", "einsteinium", "einsteinium"), ("Fm", "fermium", "fermium"), ("Md", "mendelevium", "mendélévium"),
    ("No", "nobelium", "nobélium"), ("Lr", "lawrencium", "lawrencium"), ("Rf", "rutherfordium", "rutherfordium"),
    ("Db", "dubnium", "dubnium"), ("Sg", "seaborgium", "seaborgium"), ("Bh", "bohrium", "bohrium"), ("Hs", "hassium", "hassium"),
    ("Mt", "meitnerium", "meitnérium"), ("Ds", "darmstadtium", "darmstadtium"), ("Rg", "roentgenium", "roentgenium"),
    ("Cn", "copernicium", "copernicium"), ("Nh", "nihonium", "nihonium"), ("Fl", "flerovium", "flérovium"),
    ("Mc", "moscovium", "moscovium"), ("Lv", "livermorium", "livermorium"), ("Ts", "tennessine", "tennesse"),
    ("Og", "oganesson", "oganesson")]
assert len(ELEMENTS) == 118 and len({s for s, *_ in ELEMENTS}) == 118
Z_DE = {s: z + 1 for z, (s, *_) in enumerate(ELEMENTS)}
assert (Z_DE["W"], Z_DE["Ag"], Z_DE["Au"], Z_DE["Hg"], Z_DE["Pb"], Z_DE["U"], Z_DE["Sn"], Z_DE["I"], Z_DE["Rn"],
        Z_DE["Og"]) == (74, 47, 79, 80, 82, 92, 50, 53, 86, 118)
# Pays aux capitales sans ambiguïté : (nom français, nom anglais, capitales acceptées, toutes langues confondues)
PAYS = [
    ("Australie", "Australia", ["Canberra"]), ("Canada", "Canada", ["Ottawa"]), ("Brésil", "Brazil", ["Brasilia"]),
    ("Turquie", "Turkey", ["Ankara"]), ("Nigeria", "Nigeria", ["Abuja"]), ("Pakistan", "Pakistan", ["Islamabad"]),
    ("Suisse", "Switzerland", ["Berne", "Bern"]), ("Nouvelle-Zélande", "New Zealand", ["Wellington"]),
    ("Maroc", "Morocco", ["Rabat"]), ("Viêt Nam", "Vietnam", ["Hanoi"]),
    ("Kazakhstan", "Kazakhstan", ["Astana", "Nour-Soultan", "Nur-Sultan"]),
    ("Birmanie", "Myanmar", ["Naypyidaw", "Nay Pyi Taw", "Naypyitaw"]), ("Tanzanie", "Tanzania", ["Dodoma"]),
    ("Côte d'Ivoire", "Ivory Coast", ["Yamoussoukro"]), ("Égypte", "Egypt", ["Le Caire", "Caire", "Cairo"]),
    ("Argentine", "Argentina", ["Buenos Aires"]), ("Chili", "Chile", ["Santiago", "Santiago du Chili", "Santiago de Chile"]),
    ("Pérou", "Peru", ["Lima"]), ("Colombie", "Colombia", ["Bogota"]), ("Venezuela", "Venezuela", ["Caracas"]),
    ("Équateur", "Ecuador", ["Quito"]), ("Uruguay", "Uruguay", ["Montevideo"]), ("Paraguay", "Paraguay", ["Asuncion"]),
    ("Mexique", "Mexico", ["Mexico", "Mexico City", "Ciudad de Mexico"]), ("Cuba", "Cuba", ["La Havane", "Havane", "Havana", "La Habana"]),
    ("Jamaïque", "Jamaica", ["Kingston"]), ("Kenya", "Kenya", ["Nairobi"]),
    ("Éthiopie", "Ethiopia", ["Addis-Abeba", "Addis Ababa"]), ("Ghana", "Ghana", ["Accra"]), ("Sénégal", "Senegal", ["Dakar"]),
    ("Mali", "Mali", ["Bamako"]), ("Burkina Faso", "Burkina Faso", ["Ouagadougou"]), ("Niger", "Niger", ["Niamey"]),
    ("Cameroun", "Cameroon", ["Yaoundé"]), ("Algérie", "Algeria", ["Alger", "Algiers"]), ("Tunisie", "Tunisia", ["Tunis"]),
    ("Libye", "Libya", ["Tripoli"]), ("Soudan", "Sudan", ["Khartoum"]), ("Ouganda", "Uganda", ["Kampala"]),
    ("Rwanda", "Rwanda", ["Kigali"]), ("Madagascar", "Madagascar", ["Antananarivo", "Tananarive"]),
    ("Mozambique", "Mozambique", ["Maputo"]), ("Zimbabwe", "Zimbabwe", ["Harare"]), ("Angola", "Angola", ["Luanda"]),
    ("Iran", "Iran", ["Téhéran", "Tehran"]), ("Irak", "Iraq", ["Bagdad", "Baghdad"]),
    ("Arabie saoudite", "Saudi Arabia", ["Riyad", "Riyadh"]), ("Syrie", "Syria", ["Damas", "Damascus"]),
    ("Jordanie", "Jordan", ["Amman"]), ("Liban", "Lebanon", ["Beyrouth", "Beirut"]),
    ("Afghanistan", "Afghanistan", ["Kaboul", "Kabul"]), ("Inde", "India", ["New Delhi", "Nouvelle-Delhi"]),
    ("Népal", "Nepal", ["Katmandou", "Kathmandu"]), ("Bangladesh", "Bangladesh", ["Dacca", "Dhaka"]),
    ("Thaïlande", "Thailand", ["Bangkok"]), ("Cambodge", "Cambodia", ["Phnom Penh"]), ("Laos", "Laos", ["Vientiane"]),
    ("Malaisie", "Malaysia", ["Kuala Lumpur"]), ("Philippines", "Philippines", ["Manille", "Manila"]),
    ("Chine", "China", ["Pékin", "Beijing", "Peking"]), ("Corée du Sud", "South Korea", ["Séoul"]),
    ("Corée du Nord", "North Korea", ["Pyongyang"]),
    ("Mongolie", "Mongolia", ["Oulan-Bator", "Oulan-Baatar", "Ulaanbaatar", "Ulan Bator"]), ("Russie", "Russia", ["Moscou", "Moscow"]),
    ("Ukraine", "Ukraine", ["Kiev", "Kyiv", "Kyïv"]), ("Pologne", "Poland", ["Varsovie", "Warsaw"]),
    ("Allemagne", "Germany", ["Berlin"]), ("Espagne", "Spain", ["Madrid"]), ("Portugal", "Portugal", ["Lisbonne", "Lisbon"]),
    ("Italie", "Italy", ["Rome"]), ("Grèce", "Greece", ["Athènes", "Athens"]), ("Autriche", "Austria", ["Vienne", "Vienna"]),
    ("Hongrie", "Hungary", ["Budapest"]), ("Roumanie", "Romania", ["Bucarest", "Bucharest"]), ("Bulgarie", "Bulgaria", ["Sofia"]),
    ("Serbie", "Serbia", ["Belgrade"]), ("Croatie", "Croatia", ["Zagreb"]), ("Slovénie", "Slovenia", ["Ljubljana"]),
    ("Slovaquie", "Slovakia", ["Bratislava"]), ("Tchéquie", "Czech Republic", ["Prague", "Praha"]),
    ("Danemark", "Denmark", ["Copenhague", "Copenhagen"]), ("Norvège", "Norway", ["Oslo"]), ("Suède", "Sweden", ["Stockholm"]),
    ("Finlande", "Finland", ["Helsinki"]), ("Islande", "Iceland", ["Reykjavik"]), ("Irlande", "Ireland", ["Dublin"]),
    ("Belgique", "Belgium", ["Bruxelles", "Brussels"]), ("Estonie", "Estonia", ["Tallinn"]), ("Lettonie", "Latvia", ["Riga"]),
    ("Lituanie", "Lithuania", ["Vilnius"]), ("Biélorussie", "Belarus", ["Minsk"]), ("Arménie", "Armenia", ["Erevan", "Yerevan"]),
    ("Azerbaïdjan", "Azerbaijan", ["Bakou", "Baku"]), ("Ouzbékistan", "Uzbekistan", ["Tachkent", "Tashkent"]),
    ("Kirghizistan", "Kyrgyzstan", ["Bichkek", "Bishkek"]), ("Tadjikistan", "Tajikistan", ["Douchanbé", "Dushanbe"]),
    ("Turkménistan", "Turkmenistan", ["Achgabat", "Ashgabat", "Achkhabad", "Ashkhabad"]),
    ("Royaume-Uni", "United Kingdom", ["Londres", "London"]), ("France", "France", ["Paris"]), ("Haïti", "Haiti", ["Port-au-Prince"]),
    ("Honduras", "Honduras", ["Tegucigalpa"]), ("Nicaragua", "Nicaragua", ["Managua"]), ("Costa Rica", "Costa Rica", ["San José"]),
    ("Qatar", "Qatar", ["Doha"]), ("Oman", "Oman", ["Mascate", "Muscat"]),
    ("Émirats arabes unis", "United Arab Emirates", ["Abou Dabi", "Abu Dhabi", "Abou Dhabi"])]
assert len({p for p, *_ in PAYS}) == len(PAYS)


def norm(s):
    """Minuscules, sans accents, tirets et apostrophes en espaces."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.replace("-", " ").replace("'", " ").replace("’", " ").split())


EXEMPLES = {
    "fr": ["Élément : fer\nNuméro atomique : 26. Le fer est un métal de transition, principal constituant de l'acier.\n",
           "Élément : sodium\nSymbole chimique : Na. Ce symbole vient du latin natrium ; le sodium est un métal alcalin très réactif.\n",
           "Pays : Japon\nCapitale : Tokyo. Tokyo est la plus grande agglomération du pays et le siège du gouvernement.\n"],
    "en": ["Element: iron\nAtomic number: 26. Iron is a transition metal and the main component of steel.\n",
           "Element: sodium\nChemical symbol: Na. The symbol comes from the Latin natrium, and sodium is a highly reactive alkali metal.\n",
           "Country: Japan\nCapital: Tokyo. Tokyo is the largest metropolitan area in the country and the seat of government.\n"]}
GABARIT = {("Z", "fr"): "Élément : {s}\nNuméro atomique :", ("Z", "en"): "Element: {s}\nAtomic number:",
           ("sym", "fr"): "Élément : {s}\nSymbole chimique :", ("sym", "en"): "Element: {s}\nChemical symbol:",
           ("cap", "fr"): "Pays : {s}\nCapitale :", ("cap", "en"): "Country: {s}\nCapital:"}
LECTURE = {"Z": re.compile(r" (\d{1,3})\."), "sym": re.compile(r" ([A-Z][a-z]{0,2})\."), "cap": re.compile(r" ([^.\n]{1,40}?)\.")}
QUESTIONS = []                                   # (domaine, langue, sujet, indice de l'élément ou du pays)
for lang in ("fr", "en"):
    for z, (sym, en, fr) in enumerate(ELEMENTS):
        if sym in ("Fe", "Na"):
            continue
        for dom in ("Z", "sym"):
            QUESTIONS.append((dom, lang, fr if lang == "fr" else en, z))
    for i, (pf, pe, _) in enumerate(PAYS):
        QUESTIONS.append(("cap", lang, pf if lang == "fr" else pe, i))


def juste(q, v):
    dom, _, _, i = q
    if dom == "Z":
        return v == str(i + 1)
    if dom == "sym":
        return v == ELEMENTS[i][0]
    return norm(re.split(r"[,(]", v)[0]) in {norm(a) for a in PAYS[i][2]}


def prompt_de(q):
    dom, lang, s, _ = q
    return ("" if SMOKE_TEST else "\n".join(EXEMPLES[lang]) + "\n") + GABARIT[(dom, lang)].format(s=s)


TOUTES = list(QUESTIONS)
if SMOKE_TEST:
    rng_q = np.random.default_rng(SEED + 5)
    QUESTIONS = [QUESTIONS[i] for i in sorted(rng_q.choice(len(QUESTIONS), int(os.environ.get("SMOKE_N", "120")), replace=False))]
PROMPTS = [prompt_de(q) for q in QUESTIONS]
DOMS = np.array([q[0] for q in QUESTIONS])
print(f"[QUESTIONS] {len(QUESTIONS)} questions : " + ", ".join(f"{d} {np.sum(DOMS == d)}" for d in ("Z", "sym", "cap"))
      + f" ; {len(PAYS)} pays")
print(f"[QUESTIONS] exemple de prompt :\n{PROMPTS[0]}…")

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
    """Bruit de Gumbel par ligne, graine explicite par ligne (mêmes nombres pour toutes les branches d'un problème)."""
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


def ols(x, y):
    return np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), y, rcond=None)[0]


def regle_e1(T, ds, boots, marge, plancher_ok, interpretable=True):
    """T = a + b·Δs ; a = effet de l'erreur à surprise égale ; marge absolue (0,10 × H(C) moyen)."""
    n = len(T)
    if n < N_MIN:
        return {"n": n, "verdict": "NON TESTABLE (trop peu de questions)"}
    a, b = ols(ds, T)
    bs = np.array([ols(ds[ix], T[ix]) for ix in boots])
    aci, bci = ci(bs[:, 0]), ci(bs[:, 1])
    couv = (float(np.mean(ds < 0)), float(np.mean(ds > 0)))
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif not plancher_ok:
        v = "NON TESTABLE (plancher : explication presque déterministe)"
    elif min(couv) < COUV:
        v = "NON TESTABLE (Δs ne couvre pas 0 : extrapolation)"
    elif -marge <= aci[0] and aci[1] <= marge:
        v = "ABSENTE"
    elif aci[0] > 0:
        v = "RÉACTION À L'ERREUR"
    elif aci[1] < 0:
        v = "ENGAGEMENT (moins d'incertitude après l'erreur)"
    else:
        v = "INDÉTERMINÉ"
    return {"n": n, "a": float(a), "a_ci": aci, "b": float(b), "b_ci": bci, "couverture": couv, "marge": float(marge),
            "verdict": v}


def moyenne_ci(x, rng):
    return float(x.mean()), ci([x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT_N)])


def regle_e2(T, ds, rng, interpretable=True):
    """Questions où l'erreur est le choix le plus probable (Δs < 0) : moyenne de T."""
    x = T[ds < 0]
    if len(x) < N_SOUS:
        return {"n": int(len(x)), "verdict": "NON TESTABLE (trop peu de questions où l'erreur est la plus probable)"}
    m, c = moyenne_ci(x, rng)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > 0:
        v = "L'ERREUR L'EMPORTE"
    elif c[1] < 0:
        v = "LA SURPRISE L'EMPORTE"
    else:
        v = "INDÉTERMINÉ"
    return {"n": int(len(x)), "m": m, "ci": c, "verdict": v}


def regle_nul(x, boots, interpretable=True):
    c = ci([x[ix].mean() for ix in boots])
    v = "NON INTERPRÉTABLE" if not interpretable else ("SANS BIAIS" if c[0] <= 0 <= c[1] else "BIAISÉ")
    return {"m": float(x.mean()), "ci": c, "verdict": v}


def synthese(e1, e2, interpretable=True):
    """e1, e2 : dict modèle → verdict."""
    ms = list(e1)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    oui = [m for m in ms if e1[m] == "RÉACTION À L'ERREUR" and e2.get(m) != "LA SURPRISE L'EMPORTE"]
    non = [m for m in ms if e1[m] in ("ABSENTE", "ENGAGEMENT (moins d'incertitude après l'erreur)")]
    if len(oui) == len(ms):
        return "SOUTENUE"
    if len(non) == len(ms):
        return "RÉFUTÉE"
    if oui and non and len(oui) + len(non) == len(ms):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : effets synthétiques par question, de taille connue (entropie de base 0,2 → marge 0,02)
rng_pc = np.random.default_rng(SEED + 1)
Pp = 500
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
ds_pc = rng_pc.normal(0.5, 1.5, Pp)
ds_pos = rng_pc.normal(2.0, 0.5, Pp)
bruit = lambda: rng_pc.normal(0, 0.05, Pp)
_NMIN, N_MIN = N_MIN, 100
_NSOUS, N_SOUS = N_SOUS, 30
checks = [("E1 réaction", regle_e1(0.05 + 0.02 * ds_pc + bruit(), ds_pc, bt, 0.02, True)["verdict"], "RÉACTION À L'ERREUR"),
          ("E1 absente", regle_e1(0.02 * ds_pc + bruit(), ds_pc, bt, 0.02, True)["verdict"], "ABSENTE"),
          ("E1 absente sans pente", regle_e1(bruit(), ds_pc, bt, 0.02, True)["verdict"], "ABSENTE"),
          ("E1 engagement", regle_e1(-0.05 + 0.02 * ds_pc + bruit(), ds_pc, bt, 0.02, True)["verdict"],
           "ENGAGEMENT (moins d'incertitude après l'erreur)"),
          ("E1 plancher", regle_e1(0.05 + bruit(), ds_pc, bt, 0.02, False)["verdict"],
           "NON TESTABLE (plancher : explication presque déterministe)"),
          ("E1 extrapolation", regle_e1(0.05 + 0.02 * ds_pos + bruit(), ds_pos, bt, 0.02, True)["verdict"],
           "NON TESTABLE (Δs ne couvre pas 0 : extrapolation)"),
          ("E1 trop peu", regle_e1(np.zeros(50), np.zeros(50), bt, 0.02, True)["verdict"], "NON TESTABLE (trop peu de questions)"),
          ("E2 l'erreur l'emporte", regle_e2(0.03 + 0.01 * ds_pc + bruit(), ds_pc, rng_pc)["verdict"], "L'ERREUR L'EMPORTE"),
          ("E2 la surprise l'emporte", regle_e2(0.04 * ds_pc + bruit(), ds_pc, rng_pc)["verdict"], "LA SURPRISE L'EMPORTE"),
          ("nul sans biais", regle_nul((lambda x: x - x.mean())(bruit()), bt)["verdict"], "SANS BIAIS"),
          ("nul biaisé", regle_nul(0.05 + bruit(), bt)["verdict"], "BIAISÉ"),
          ("synthèse soutenue", synthese({"a": "RÉACTION À L'ERREUR", "b": "RÉACTION À L'ERREUR"},
                                         {"a": "INDÉTERMINÉ", "b": "L'ERREUR L'EMPORTE"}), "SOUTENUE"),
          ("synthèse réfutée", synthese({"a": "ABSENTE", "b": "ENGAGEMENT (moins d'incertitude après l'erreur)"},
                                        {"a": "INDÉTERMINÉ", "b": "INDÉTERMINÉ"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "RÉACTION À L'ERREUR", "b": "ABSENTE"},
                                       {"a": "L'ERREUR L'EMPORTE", "b": "INDÉTERMINÉ"}), "DÉPEND DU MODÈLE"),
          ("synthèse surprise", synthese({"a": "RÉACTION À L'ERREUR", "b": "RÉACTION À L'ERREUR"},
                                         {"a": "LA SURPRISE L'EMPORTE", "b": "INDÉTERMINÉ"}), "INDÉTERMINÉE")]
N_MIN, N_SOUS = _NMIN, _NSOUS
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<26} : {got:<56} (attendu {att})")
POSITIVE_CONTROL_OK = all(g == e for _, g, e in checks)
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON INTERPRÉTABLE'}")

# %%
# ---------- Chargement ----------
class SmokeTokenizer:
    """Tokenizer jouet par caractères, aller-retour exact (SMOKE_TEST)."""
    ALPHA = "?"                                  # fixé d'après le corpus du jouet (SMOKE_TEST)

    def __init__(self, V):
        self.V, self.pad_token_id, self.eos_token_id = V, 0, 1
        self.all_special_ids = [0, 1]

    def __len__(self):                          # jetons valides : l'alphabet (le reste du vocabulaire est masqué)
        return 2 + len(self.ALPHA)

    def encode(self, s, **kw):
        return [2 + self.ALPHA.index(c) if c in self.ALPHA else 2 + self.ALPHA.index("?") for c in s]

    def decode(self, ids, **kw):
        return "".join(self.ALPHA[i - 2] if 2 <= i < 2 + len(self.ALPHA) else "?" for i in ids)


def corpus_jouet(rng, n=4000):
    """Questions suivies de leur bonne réponse et d'une explication tirée de gabarits (harnais local)."""
    expl = {"fr": ["{s} est étudié en chimie et en physique.", "On trouve {s} dans de nombreux ouvrages.",
                   "{s} a une longue histoire.", "Les élèves apprennent vite le cas de {s}."],
            "en": ["{s} is studied in chemistry and physics.", "{s} appears in many textbooks.", "{s} has a long history.",
                   "Students quickly learn about {s}."]}
    lignes = []
    for _ in range(n):
        dom, lang, s, i = TOUTES[rng.integers(len(TOUTES))]
        rep = str(i + 1) if dom == "Z" else (ELEMENTS[i][0] if dom == "sym" else PAYS[i][2][0])
        lignes.append(GABARIT[(dom, lang)].format(s=s) + f" {rep}. " + expl[lang][rng.integers(4)].format(s=s) + "\n")
    return "\n".join(lignes)


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
    if model_id == "smoke-qwen3-texte":
        # Qwen3 minuscule entraîné par caractères sur les questions et leurs bonnes réponses : de vraies erreurs (harnais)
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512)
        torch.manual_seed(SEED)
        m, tk = Qwen3ForCausalLM(cfg).to(DEVICE), SmokeTokenizer(512)
        rng = np.random.default_rng(SEED + 99)
        texte = corpus_jouet(rng)
        SmokeTokenizer.ALPHA = "?" + "".join(sorted(set(texte + "".join(PROMPTS)) - {"?"}))
        assert len(SmokeTokenizer.ALPHA) + 2 <= 512
        corpus = np.array(tk.encode(texte))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for _ in range(int(os.environ.get("SMOKE_PAS", "1500"))):          # fenêtres de 192 caractères : tout le prompt
            d = rng.integers(0, len(corpus) - 193, 32)
            x = torch.tensor(np.stack([corpus[i:i + 192] for i in d]), device=DEVICE)
            loss = m(input_ids=x, labels=x).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m.eval(), tk, f"fp32 entraîné sur les questions (smoke, perte finale {loss.item():.2f})"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    else:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": 0}, "fp16 1 GPU"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda":
        assert next(model.parameters()).dtype == torch.float16, "type de poids inattendu"
    return model, tok, mode

# %%
# ---------- Génération (comme A1c–A1z) ----------
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
        self._dec = {}

    def dec(self, t):
        t = int(t)
        if t not in self._dec:
            self._dec[t] = self.tok.decode([t])
        return self._dec[t]

    def left_pad(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.dev), attn.to(self.dev), L

    @torch.no_grad()
    def sample(self, ctxs, prompt_of, T, nsteps, step0, base=SEED):
        """Graine du pas s de la ligne j : base + 100003·prompt_of[j] + step0 + s (nombres aléatoires communs)."""
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
            seeds = [base + 100003 * int(prompt_of[j]) + step0 + s for j in range(n)]
            score = z / T + gumbel_noise(self.V, seeds, z.device)
            score[:, self.forbid] = -float("inf")
            if self.V > self.n_tok:
                score[:, self.n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            S[:, s], tk[:, s] = e1, nxt.cpu().numpy()
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = pos[:, -1:] + 1
        del past, out
        return S, tk

# %%
# ---------- Sélection, surprise, branches ----------
@torch.no_grad()
def tirer_reponses(g, enc, K):
    """K réponses libres par question (bruit SEL, graine propre à chaque tirage). Valeur lue selon le domaine, jusqu'au point
    (None si illisible)."""
    P = len(enc)
    vals = [[None] * K for _ in range(P)]
    jobs = [(p, j) for p in range(P) for j in range(K)]
    nb = (len(jobs) + g.budget - 1) // g.budget
    for b, b0 in enumerate(range(0, len(jobs), g.budget)):
        t0 = time.time()
        lot = jobs[b0:b0 + g.budget]
        _, tk = g.sample([enc[p] for p, _ in lot], [p * 64 + j for p, j in lot], T_MAIN, LONG_REP, 0, base=SEL)
        for (p, j), row in zip(lot, tk):
            m = LECTURE[DOMS[p]].match(g.tok.decode(row.tolist()))
            vals[p][j] = m.group(1).strip() if m and m.group(1).strip() else None
        if b == 0:
            est = (time.time() - t0) * nb / 60
            print(f"[PROGRESSION] {g.label} sélection : premier lot en {time.time() - t0:.1f} s ; estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
    return vals


@torch.no_grad()
def surprise(g, enc, suff):
    """−log p (loi du modèle, T = 1) des jetons forcés suff[i] après enc[i]."""
    base, head = g.model.model, g.model.get_output_embeddings()
    out = np.zeros(len(enc))
    for b0 in range(0, len(enc), g.budget):
        loc = list(range(b0, min(len(enc), b0 + g.budget)))
        ids, attn, L = g.left_pad([enc[i] + suff[i] for i in loc])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
        nmax = max(len(suff[i]) for i in loc)
        lp = torch.log_softmax(head(h[:, L - nmax - 1:L - 1, :].to(head.weight.device)).double(), dim=-1)
        for j, i in enumerate(loc):
            n = len(suff[i])
            tgt = torch.tensor(suff[i], device=lp.device)
            out[i] = -float(lp[j, nmax - n:].gather(-1, tgt[:, None]).sum())
    return out


@torch.no_grad()
def suites(g, enc, idx, forces, base):
    """Suite libre de LONG_SUITE jetons après enc[idx[i]] + forces[i] ; graine du jeton k : base + 100003·idx + ETAPE_SUITE + k,
    donc le même bruit à la position k dans toutes les branches d'un problème. S[:, k − 1] = entropie de la loi du jeton k."""
    P = len(idx)
    S, TK = np.zeros((P, LONG_SUITE), np.float32), np.zeros((P, LONG_SUITE), np.int64)
    for b0 in range(0, P, g.budget):
        loc = list(range(b0, min(P, b0 + g.budget)))
        s, t = g.sample([enc[int(idx[i])] + forces[i] for i in loc], [int(idx[i]) for i in loc], T_MAIN, LONG_SUITE,
                        ETAPE_SUITE, base=base)
        S[loc], TK[loc] = s, t
    return S, TK


def fenetre(g, TK):
    """Nombre de positions mesurées : jusqu'au premier saut de ligne inclus, au plus KMAX."""
    L = np.full(len(TK), KMAX)
    for i, row in enumerate(TK):
        for k, t in enumerate(row[:KMAX]):
            if "\n" in g.dec(t):
                L[i] = k + 1
                break
    return L


def h_post(S, L):
    return np.array([float(S[i, :L[i]].mean()) for i in range(len(S))])

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


class TropPeu(Exception):
    pass


def ecrire_resume():
    with open(os.path.join(OUT, "resume_b1b.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def fmt(x, c, f="+.4f"):
    return f"{x:{f}} IC95 [{c[0]:{f}} ; {c[1]:{f}}]"


def element_de(dom, v):
    """Indice de l'élément désigné par une réponse (numéro ou symbole), ou None."""
    if dom == "Z" and v.isdigit() and 1 <= int(v) <= 118:
        return int(v) - 1
    if dom == "sym" and v in Z_DE:
        return Z_DE[v] - 1
    return None


AMBIGUS = {"or", "lead"}                         # noms d'éléments qui sont aussi des mots courants


def mots(s):
    return " " + re.sub(r"[^0-9a-z]+", " ", norm(s)).strip() + " "


def cite(texte, noms):
    t = mots(texte)
    return any(mots(n) in t for n in noms if n and norm(n) not in AMBIGUS)


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
        g = Gen(model, tok, label)
        enc = [tok.encode(p) for p in PROMPTS]
        a = {}

        # 1. Sélection
        t1 = time.time()
        vals = tirer_reponses(g, enc, K_ECH)
        lisible = np.array([[v is not None for v in r] for r in vals])
        exact = np.array([[v is not None and juste(QUESTIONS[i], v) for v in r] for i, r in enumerate(vals)])
        a["selection"] = {"lisible": float(lisible.mean()), "exactitude": float(exact.mean()),
                          "exactitude_par_domaine": {d: float(exact[DOMS == d].mean()) for d in ("Z", "sym", "cap")}}
        idx, FC, FW, VC, VW = [], [], [], [], []
        n_tok_ko = 0
        for i, r in enumerate(vals):
            bons = [v for v in r if v is not None and juste(QUESTIONS[i], v)]
            faux = [v for v in r if v is not None and not juste(QUESTIONS[i], v)]
            if not bons or not faux:
                continue
            plus = lambda xs: max(set(xs), key=lambda v: (xs.count(v), -xs.index(v)))
            c, w = plus(bons), plus(faux)
            fc, fw = tok.encode(PROMPTS[i] + f" {c}."), tok.encode(PROMPTS[i] + f" {w}.")
            if fc[:len(enc[i])] != enc[i] or fw[:len(enc[i])] != enc[i]:
                n_tok_ko += 1
                continue
            idx.append(i)
            FC.append(fc[len(enc[i]):])
            FW.append(fw[len(enc[i]):])
            VC.append(c)
            VW.append(w)
        idx = np.array(idx, np.int64)
        P = len(idx)
        a["selection"].update({"retenus": P, "jetons_incoherents": n_tok_ko,
                               "retenus_par_domaine": {d: int(np.sum(DOMS[idx] == d)) for d in ("Z", "sym", "cap")} if P else {}})
        print(f"[SÉLECTION] {label} : réponses lisibles {lisible.mean():.1%}, exactes {exact.mean():.1%} ("
              + ", ".join(f"{d} {exact[DOMS == d].mean():.1%}" for d in ("Z", "sym", "cap")) + f") ; questions retenues "
              f"(au moins une bonne et une mauvaise réponse) : {P} sur {len(QUESTIONS)} ("
              + ", ".join(f"{d} {np.sum(DOMS[idx] == d)}" for d in ("Z", "sym", "cap")) + f") ; découpage incohérent {n_tok_ko} ; "
              f"{time.time() - t1:.0f} s", flush=True)
        if P < N_MIN:
            a["E1"] = {"n": P, "verdict": "NON TESTABLE (trop peu de questions)"}
            a["E2"] = {"n": 0, "verdict": "NON TESTABLE (trop peu de questions)"}
            ALL[label] = a
            print(f"[E1] {label} : {a['E1']['verdict']}")
            raise TropPeu
        if n_tok_ko > 0.1 * (P + n_tok_ko):
            raise RuntimeError("le découpage en jetons de « v. » ne prolonge pas celui du prompt : réponses forcées non fiables")

        # 2. Surprise des deux réponses
        s_c = surprise(g, [enc[i] for i in idx], FC)
        s_w = surprise(g, [enc[i] for i in idx], FW)
        ds = s_w - s_c

        # 3. Branches
        t3 = time.time()
        S2, TK2 = suites(g, enc, np.repeat(idx, 2), [f for pr in zip(FC, FW) for f in pr], SEED)
        S_C, S_W, TK_C, TK_W = S2[0::2], S2[1::2], TK2[0::2], TK2[1::2]
        S_R, TK_R = suites(g, enc, idx, FC, SEED)
        S_N, TK_N = suites(g, enc, idx, FC, FRAIS)
        L_C, L_W, L_N = fenetre(g, TK_C), fenetre(g, TK_W), fenetre(g, TK_N)
        same_1 = float(np.mean(TK_R[:, 0] == TK_C[:, 0]))
        same = float(np.mean([np.array_equal(TK_R[i, :L_C[i]], TK_C[i, :L_C[i]]) for i in range(P)]))
        dS = float(np.abs(S_R[:, 0] - S_C[:, 0]).max())
        print(f"[VÉRIF] {label} reprise (C régénérée dans d'autres lots) : premier jeton identique {same_1:.1%}, explication entière "
              f"identique {same:.1%}, écart max d'entropie au jeton 1 {dS:.1e} ; longueur médiane de la fenêtre C {np.median(L_C):.0f}, "
              f"W {np.median(L_W):.0f} ; branches : {time.time() - t3:.0f} s", flush=True)
        if same_1 < 0.9 or same < 0.2 or dS > 0.05:
            raise RuntimeError("la régénération ne redonne pas la branche C : chemin de génération défectueux")
        a["reprise"] = {"premier": same_1, "explication": same, "ecart_S1": dS}

        # 4. Mesures et règles
        H_C, H_W, H_N = h_post(S_C, L_C), h_post(S_W, L_W), h_post(S_N, L_N)
        T, T_nul = H_W - H_C, H_N - H_C
        med_C = float(np.median(H_C))
        plancher_ok = med_C >= PLANCHER
        marge = MARGE_REL * float(H_C.mean())
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        cim = lambda x: ci([x[ix].mean() for ix in boots])
        nul = regle_nul(T_nul, boots, ok)
        t_ok = ok and nul["verdict"] == "SANS BIAIS"
        e1 = regle_e1(T, ds, boots, marge, plancher_ok, t_ok)
        e2 = regle_e2(T, ds, np.random.default_rng(SEED + 4), t_ok and plancher_ok) if plancher_ok else \
            {"n": int(np.sum(ds < 0)), "verdict": "NON TESTABLE (plancher : explication presque déterministe)"}
        app = np.abs(ds) <= ECART_APP
        e3 = moyenne_ci(T[app], np.random.default_rng(SEED + 5)) if app.sum() >= N_SOUS else (float("nan"), (float("nan"),) * 2)
        rk = lambda x: np.argsort(np.argsort(x))
        rho = float(np.corrcoef(rk(ds), rk(T))[0, 1])
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN}) : {P} questions")
        print(f"[PLANCHER] {label} entropie de l'explication après la bonne réponse : médiane {med_C:.4f}, moyenne {H_C.mean():.4f} "
              f"(seuil {PLANCHER}) → {'franchi' if plancher_ok else 'NON FRANCHI : E1 et E2 NON TESTABLES'} ; marge d'équivalence "
              f"±{marge:.4f}")
        print(f"[SURPRISE] {label} : s(C) médiane {np.median(s_c):.2f} nats, s(W) médiane {np.median(s_w):.2f} nats ; Δs médian "
              f"{np.median(ds):+.2f} ; erreur plus probable que la bonne réponse (Δs < 0) : {np.mean(ds < 0):.0%}")
        print(f"[MESURE] {label} : H(C) = {H_C.mean():.4f}, H(W) = {H_W.mean():.4f} ; T = H(W) − H(C) = {fmt(T.mean(), cim(T))} ; "
              f"Spearman(Δs, T) {rho:+.3f} ; T > 0 dans {np.mean(T > 0):.1%} des questions")
        print(f"[TÉMOIN NÉGATIF] {label} T(nul) (bonne réponse, autre bruit) = {fmt(nul['m'], nul['ci'])} → {nul['verdict']}"
              + ("" if t_ok or not ok else " : E1 et E2 NON INTERPRÉTABLES"))
        if "a" in e1:
            print(f"[E1] {label} T = a + b·Δs : a = {fmt(e1['a'], e1['a_ci'])} (marge ±{marge:.4f}) ; b = {fmt(e1['b'], e1['b_ci'])} "
                  f"par nat ; Δs < 0 : {e1['couverture'][0]:.0%}, Δs > 0 : {e1['couverture'][1]:.0%} → {e1['verdict']}")
        else:
            print(f"[E1] {label} : {e1['verdict']}")
        print(f"[E2] {label} erreur plus probable que la bonne réponse ({e2['n']} questions) : "
              + (f"T = {fmt(e2['m'], e2['ci'])} → " if "m" in e2 else "") + e2["verdict"])
        print(f"[APPARIÉ] {label} |Δs| ≤ {ECART_APP} ({int(app.sum())} questions) : T = {fmt(e3[0], e3[1])}")
        par_dom = {}
        for d in ("Z", "sym", "cap"):
            msk = DOMS[idx] == d
            if msk.sum() >= 10:
                m_, c_ = moyenne_ci(T[msk], np.random.default_rng(SEED + 6))
                par_dom[d] = {"n": int(msk.sum()), "T": m_, "ci": c_}
                print(f"[DOMAINE] {label} {d:<4} ({msk.sum()} questions) : T = {fmt(m_, c_)}")
        txt_W = [tok.decode(TK_W[i, :L_W[i]].tolist()) for i in range(P)]
        txt_C = [tok.decode(TK_C[i, :L_C[i]].tolist()) for i in range(P)]
        corr, prop_w, cite_q = [], [], []
        for j, i in enumerate(idx):
            dom, lang, s, k = QUESTIONS[i]
            if dom == "sym":                     # symboles : sensible à la casse, mot entier
                corr.append(bool(re.search(r"(?<![A-Za-z])" + re.escape(VC[j]) + r"(?![a-z])", txt_W[j])))
            else:
                corr.append(cite(txt_W[j], [VC[j]] + (list(PAYS[k][2]) if dom == "cap" else [])))
            if dom in ("Z", "sym"):
                ew = element_de(dom, VW[j])
                prop_w.append(ew is not None and ew != k and cite(txt_W[j], ELEMENTS[ew][1:]))
                cite_q.append(cite(txt_W[j], ELEMENTS[k][1:]))
        a["explications"] = {"correction": float(np.mean(corr)), "parle_element_mauvaise_reponse": float(np.mean(prop_w)) if prop_w else None,
                             "parle_element_demande": float(np.mean(cite_q)) if cite_q else None}
        print(f"[EXPLICATION] {label} après W : cite la bonne réponse {np.mean(corr):.1%} ; numéros et symboles ({len(prop_w)}) : parle de "
              f"l'élément de la mauvaise réponse {np.mean(prop_w) if prop_w else float('nan'):.1%}, de l'élément demandé "
              f"{np.mean(cite_q) if cite_q else float('nan'):.1%}")
        for j in list(np.argsort(-np.abs(T))[:3]):
            dom, lang, s, k = QUESTIONS[idx[j]]
            print(f"[EXEMPLE] {label} {dom} « {s} » : C « {VC[j]} » → « {txt_C[j].strip()[:110]} » | W « {VW[j]} » → "
                  f"« {txt_W[j].strip()[:110]} » (T {T[j]:+.3f})")
        prof = [float((S_W - S_C)[:, lo - 1:hi].mean()) for lo, hi in TRANCHES]
        print(f"[PROFIL] {label} W − C par tranches de k ({' | '.join(f'{lo}–{hi}' for lo, hi in TRANCHES)}) : "
              + "  ".join(f"{x:+.4f}" for x in prof))
        a.update({"n": P, "plancher": {"mediane_HC": med_C, "franchi": plancher_ok}, "marge": marge,
                  "s_c_med": float(np.median(s_c)), "s_w_med": float(np.median(s_w)), "ds_neg": float(np.mean(ds < 0)),
                  "H_C": float(H_C.mean()), "H_W": float(H_W.mean()), "T": float(T.mean()), "T_ci": cim(T),
                  "part_T_pos": float(np.mean(T > 0)), "spearman_ds_T": rho, "temoin_negatif": nul, "E1": e1, "E2": e2,
                  "apparie": {"n": int(app.sum()), "T": e3[0], "ci": e3[1]}, "par_domaine": par_dom, "profil": prof})
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_b1b.npz"), idx=idx, s_c=s_c, s_w=s_w, S_C=S_C, S_W=S_W, S_N=S_N,
                            TK_C=TK_C, TK_W=TK_W, TK_N=TK_N, L_C=L_C, L_W=L_W, L_N=L_N, VC=np.array(VC), VW=np.array(VW))
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min", flush=True)
    except TropPeu:
        pass
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] B1b — après une erreur factuelle, l'explication devient-elle plus incertaine, au-delà de la surprise ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "E1" in ALL[m]]
v = {k: {m: ALL[m][k]["verdict"] for m in labs} for k in ("E1", "E2")}
for m in labs:
    print(f"[SYNTHÈSE] {m:<22} : E1 {v['E1'][m]:<56} ; E2 {v['E2'][m]}")
these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese(v["E1"], v["E2"], ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « le modèle réagit à sa propre erreur au-delà de la surprise » : {these} "
      f"(prédiction SOUTENUE 20 %, RÉFUTÉE 30 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

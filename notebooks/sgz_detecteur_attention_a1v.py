# %% [markdown]
# # SGZ A1v — L'attention reçue repère-t-elle un jeton inséré sans connaître sa position ?
#
# **Point de départ (A1r, après coup, 28 oct.).** Quand on sait où chercher, l'attention reçue par le jeton anormal le
# trahit très bien : AUC 0,93 à 0,996 au pas suivant, contre 0,83 à 0,90 pour l'entropie (Qwen3-1.7B-Base, SmolLM3-3B-Base).
# Mais un vrai détecteur ne connaît pas la position. Et la référence la plus simple, la surprise du jeton lui-même, n'avait
# jamais été mesurée : c'est elle qu'il faut battre.
#
# **Dispositif.** Témoin généré (768 prompts, T = 0,6, 160 jetons, même graine que A1c–A1t). Pour chaque prompt, une
# position t₀ tirée au hasard entre 40 et 128 ; deux textes « injectés » : la suite du témoin où seul le jeton t₀ est
# remplacé par un jeton **plausible** (rang 20–100 dans la loi du témoin) ou **rare** (rang 1000–5000), jamais celui du
# témoin. Passes rejouées (forçage par l'enseignant) des trois textes ; pour chaque jeton i généré : attention reçue au pas
# suivant r₁(i) (moyenne sur couches et têtes de l'attention de la position i + 1 vers i), attention reçue sur huit pas
# r₈(i), entropie de Tsallis (q = 1,45) de la loi suivante, surprise −log p du jeton. Zone de recherche : jetons générés
# 8 à 151 (le détecteur ne sait pas où est t₀). Deux modèles : Qwen3-1.7B-Base et SmolLM3-3B-Base.
#
# ## Critères (fixés avant toute exécution ; AUC de Mann-Whitney, texte injecté contre témoin, IC95 par bootstrap sur
# les prompts, les deux textes d'un prompt tirés ensemble, B = 1000)
#
# Règle de détection : DÉTECTABLE si borne basse ≥ 0,75 ; FAIBLE si borne basse > 0,5 ; INVERSE si borne haute < 0,5 ;
# sinon NON DÉTECTABLE.
# Score principal : le **contraste local** c₁(i) = r₁(i) − médiane de r₁ sur les 8 jetons de part et d'autre (i exclu).
# Raison (vue sur le harnais, avant toute donnée réelle) : l'attention reçue baisse avec la longueur du contexte, et un
# maximum brut tombe sur les premières positions de la zone quelle que soit l'anomalie. Le maximum brut de r₁ reste en
# descriptif.
# - **V1 (principal)** : score = max de c₁ sur la zone, jeton rare.
# - **V2** : même score, jeton plausible.
# - **V3, localisation** (textes injectés, jeton rare) : position prédite = argmax de c₁ ; LOCALISABLE si la borne basse de
#   la part exacte > 0,5 ; PARTIELLE si elle dépasse 5 × le hasard (1 / 144) ; sinon NON LOCALISABLE.
# - **V4, contre la référence** : ΔAUC = AUC(max c₁) − AUC(max de la surprise), pour chaque type : BAT LA SURPRISE si
#   IC95 > 0 ; ÉGALE si IC95 ⊂ [−0,02 ; +0,02] ; MOINS BONNE si IC95 < 0 ; sinon INDÉTERMINÉ.
# - **Synthèse, thèse « l'attention reçue repère un jeton inséré sans connaître sa position »** (sur V1, contraste local) : SOUTENUE si
#   DÉTECTABLE pour les deux modèles ; PARTIELLE si les deux sont au moins FAIBLE sans être DÉTECTABLE deux fois ; RÉFUTÉE si
#   les deux sont NON DÉTECTABLE ou INVERSE ; DÉPEND DU MODÈLE si l'un est DÉTECTABLE et l'autre NON DÉTECTABLE ou INVERSE ;
#   sinon INDÉTERMINÉE.
# - **Descriptifs** : AUC de max r₁ brut, max r₈, max de l'entropie, saut maximal d'entropie, max de la surprise ; localisation par la
#   surprise et par r₈ ; AUC à position connue (r₁ en t₀) pour retrouver A1r ; surprise médiane des jetons injectés et
#   distribution de la surprise maximale des témoins.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1 DÉTECTABLE 55 %, FAIBLE 35 %. V2 FAIBLE 55 %, NON DÉTECTABLE 30 %. V3 PARTIELLE 40 %, LOCALISABLE 35 %. V4 jeton rare :
# MOINS BONNE que la surprise 60 %, ÉGALE 25 % ; jeton plausible : BAT LA SURPRISE 30 %. Synthèse SOUTENUE 45 %.
#
# ## Limites connues
#
# - Anomalies artificielles (un jeton remplacé), suite inchangée : c'est le cas d'une insertion dans un texte existant,
#   pas celui d'une erreur que le modèle aurait produite lui-même.
# - Attention moyennée sur toutes les têtes ; un détecteur fondé sur quelques têtes choisies pourrait faire mieux (non testé).
# - Les jetons qui reçoivent naturellement beaucoup d'attention (ponctuation, retours à la ligne) concurrencent l'anomalie :
#   c'est précisément ce que mesure la position inconnue.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~10 min pour Qwen3,
# ~15 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[V1]` à `[V4]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1v" if os.path.isdir("/kaggle/working") else "sgz_a1v_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
Q = 1.45
N_PROMPTS, GEN_LEN = 768, 160
T0_LO, T0_HI = 40, 128                          # position de l'insertion, tirée par prompt
ZONE = (8, 151)                                 # jetons générés examinés par le détecteur (r₈ complet jusqu'à 151)
TYPES = {"plausible": (20, 100), "rare": (1000, 5000)}
K8 = 8
EQUIV = 0.02
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                 # même graine que A1c–A1t
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN = 16, 64
    T0_LO, T0_HI = 16, 40
    ZONE = (4, 54)
    TYPES = {"plausible": (3, 8), "rare": (40, 80)}
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-aleatoire-bis", ["smoke-qwen3-random"])]
BRAS = ["temoin"] + list(TYPES)
assert ZONE[0] < T0_LO and T0_HI < ZONE[1] and ZONE[1] + K8 <= GEN_LEN - 1
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀ ∈ [{T0_LO} ; {T0_HI}] zone {ZONE} ; types {TYPES} ; "
      f"modèles={[m[0] for m in MODELS]}")
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


from scipy.stats import rankdata


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def regle_auc(c, interpretable=True):
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if c[0] >= 0.75:
        return "DÉTECTABLE"
    if c[0] > 0.5:
        return "FAIBLE"
    if c[1] < 0.5:
        return "INVERSE"
    return "NON DÉTECTABLE"


def juger_auc(sp, sn, boots, interpretable=True):
    a = auc(sp, sn)
    c = ci([auc(sp[ix], sn[ix]) for ix in boots])
    return {"auc": a, "ci": c, "verdict": regle_auc(c, interpretable)}


def regle_loc(exact, boots, n_pos, interpretable=True):
    c = ci([exact[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > 0.5:
        v = "LOCALISABLE"
    elif c[0] > 5 / n_pos:
        v = "PARTIELLE"
    else:
        v = "NON LOCALISABLE"
    return {"part": float(exact.mean()), "ci": c, "verdict": v}


def regle_delta(sa_p, sa_n, sb_p, sb_n, boots, interpretable=True):
    """ΔAUC = AUC(score a) − AUC(score b), mêmes rééchantillons."""
    d = auc(sa_p, sa_n) - auc(sb_p, sb_n)
    c = ci([auc(sa_p[ix], sa_n[ix]) - auc(sb_p[ix], sb_n[ix]) for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] >= -EQUIV and c[1] <= EQUIV:
        v = "ÉGALE"
    elif c[0] > 0:
        v = "BAT LA SURPRISE"
    elif c[1] < 0:
        v = "MOINS BONNE"
    else:
        v = "INDÉTERMINÉ"
    return {"delta": float(d), "ci": c, "verdict": v}


def synthese(v1, interpretable=True):
    vs = list(v1.values())
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not vs:
        return "INDÉTERMINÉE"
    if all(v == "DÉTECTABLE" for v in vs):
        return "SOUTENUE"
    if all(v in ("DÉTECTABLE", "FAIBLE") for v in vs):
        return "PARTIELLE"
    if all(v in ("NON DÉTECTABLE", "INVERSE") for v in vs):
        return "RÉFUTÉE"
    if "DÉTECTABLE" in vs and any(v in ("NON DÉTECTABLE", "INVERSE") for v in vs):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : scores synthétiques par prompt dont l'AUC vraie est connue (décalage gaussien d = √2 Φ⁻¹(AUC))
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]


def scores(d):
    return rng_pc.normal(d, 1, Pp), rng_pc.normal(0, 1, Pp)


p9, n9 = scores(2.0)                # AUC vraie ≈ 0,92
p6, n6 = scores(0.6)                # ≈ 0,66
p5, n5 = scores(0.0)                # 0,5
pi_, ni_ = scores(-0.6)             # ≈ 0,34
checks = [("détectable", juger_auc(p9, n9, bt)["verdict"], "DÉTECTABLE"),
          ("faible", juger_auc(p6, n6, bt)["verdict"], "FAIBLE"),
          ("non détectable", juger_auc(p5, n5, bt)["verdict"], "NON DÉTECTABLE"),
          ("inverse", juger_auc(pi_, ni_, bt)["verdict"], "INVERSE"),
          ("localisable", regle_loc((rng_pc.random(Pp) < 0.8).astype(float), bt, 144)["verdict"], "LOCALISABLE"),
          ("localisation partielle", regle_loc((rng_pc.random(Pp) < 0.2).astype(float), bt, 144)["verdict"], "PARTIELLE"),
          ("localisation nulle", regle_loc((rng_pc.random(Pp) < 1 / 144).astype(float), bt, 144)["verdict"], "NON LOCALISABLE"),
          ("bat la surprise", regle_delta(p9, n9, p6, n6, bt)["verdict"], "BAT LA SURPRISE"),
          ("moins bonne", regle_delta(p6, n6, p9, n9, bt)["verdict"], "MOINS BONNE"),
          ("égale", regle_delta(p9, n9, p9 + 1e-6, n9, bt)["verdict"], "ÉGALE"),
          ("synthèse soutenue", synthese({"a": "DÉTECTABLE", "b": "DÉTECTABLE"}), "SOUTENUE"),
          ("synthèse partielle", synthese({"a": "DÉTECTABLE", "b": "FAIBLE"}), "PARTIELLE"),
          ("synthèse réfutée", synthese({"a": "NON DÉTECTABLE", "b": "INVERSE"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "DÉTECTABLE", "b": "NON DÉTECTABLE"}), "DÉPEND DU MODÈLE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<24} : {got:<18} (attendu {att})")
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
# ---------- Écriture des jetons, choix des quatre bras ----------
def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque


@torch.no_grad()
def generer_temoin(g, enc):
    """Témoin seul (même graine par prompt que A1c–A1t) ; au pas t₀ propre à chaque prompt, choix des jetons insérés."""
    P = len(enc)
    rng = np.random.default_rng(SEED + 21)
    t0s = rng.integers(T0_LO, T0_HI + 1, P)
    cibles = {t: rng.integers(lo, hi + 1, P) for t, (lo, hi) in TYPES.items()}
    tokm = np.zeros((P, GEN_LEN), np.int64)
    choix = {t: {"jeton": np.zeros(P, np.int64), "rang": np.zeros(P, np.int64), "surp": np.zeros(P)} for t in TYPES}
    npb = g.budget
    n_lots = math.ceil(P / npb)
    t_debut = time.time()
    for b0 in range(0, P, npb):
        tl = time.time()
        rows = list(range(b0, min(P, b0 + npb)))

        def force(s, z, nxt, rows=rows):
            js = [j for j, r in enumerate(rows) if t0s[r] == s]
            if js:
                zz = g.masque(z[js].clone())
                lp = torch.log_softmax(z[js].double(), dim=-1)
                order = torch.argsort(zz, dim=-1, descending=True)
                for jj, j in enumerate(js):
                    r = rows[j]
                    for t in TYPES:
                        rk = int(cibles[t][r])
                        if int(order[jj, rk]) == int(nxt[j]):        # jamais le jeton du témoin
                            rk += 1
                        x = int(order[jj, rk])
                        choix[t]["jeton"][r], choix[t]["rang"][r], choix[t]["surp"][r] = x, rk, -float(lp[jj, x])
            return nxt

        _, tk = g.sample([enc[r] for r in rows], rows, T_MAIN[0], GEN_LEN, 0, force)
        tokm[rows] = tk
        g.progress(time.time() - tl, n_lots)
    print(f"[MESURE] {g.label} génération du témoin : {time.time() - t_debut:.0f} s", flush=True)
    return tokm, t0s, choix


@torch.no_grad()
def rejouer(g, ctxs, conts):
    """Pour chaque texte imposé (jetons générés 0 … n − 1) : entropie de la loi du jeton i, surprise du jeton i, attention
    reçue par le jeton i au pas suivant (r₁) et sur huit pas (r₈), moyennes sur couches et têtes."""
    base, head = g.model.model, g.model.get_output_embeddings()
    n, nc = len(ctxs), len(conts[0])
    ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    out = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False, output_attentions=True)
    att = out.attentions
    assert att is not None and att[0] is not None, "poids d'attention indisponibles"
    M = None
    for a in att:                                          # moyenne sur têtes puis couches : (n, T, T)
        m = a.float().mean(1).to(ids.device)
        M = m if M is None else M + m
    M = M / len(att)
    p0 = L - nc                                            # position du jeton généré 0
    idx = torch.arange(nc, device=ids.device)
    r1 = torch.full((n, nc), float("nan"), device=ids.device)
    r1[:, :nc - 1] = M[:, p0 + idx[:-1] + 1, p0 + idx[:-1]]
    r8 = torch.full((n, nc), float("nan"), device=ids.device)
    lim = nc - K8
    r8[:, :lim] = torch.stack([M[:, p0 + idx[:lim] + d, p0 + idx[:lim]] for d in range(1, K8 + 1)]).mean(0)
    haut = float(torch.triu(M[:, p0:, p0:], diagonal=1).abs().max())    # masque causal : doit être 0
    hk = out.last_hidden_state[:, p0 - 1:L - 1, :].reshape(n * nc, -1)  # la position p prédit le jeton p + 1
    cible = torch.tensor([x for s in conts for x in s], device=head.weight.device)
    S, sp = np.zeros(n * nc), np.zeros(n * nc)
    for c0 in range(0, n * nc, 192):
        z = head(hk[c0:c0 + 192].to(head.weight.device)).float()
        S[c0:c0 + 192] = entropies_from_logits(z, [Q])[Q].cpu().numpy()
        lp = torch.log_softmax(z.double(), dim=-1)
        sp[c0:c0 + 192] = (-lp.gather(1, cible[c0:c0 + 192, None])[:, 0]).cpu().numpy()
    del out, att, M
    return (S.reshape(n, nc).astype(np.float32), sp.reshape(n, nc).astype(np.float32),
            r1.cpu().numpy().astype(np.float32), r8.cpu().numpy().astype(np.float32), haut)


@torch.no_grad()
def run_replay(g, enc, tokm, t0s, choix):
    P = len(enc)
    R = {a: {k: np.zeros((P, GEN_LEN), np.float32) for k in ("S", "surp", "r1", "r8")} for a in BRAS}
    nb, haut_max = 8, 0.0
    t_debut = time.time()
    for b0 in range(0, P, nb):
        rows = list(range(b0, min(P, b0 + nb)))
        ct = [tokm[r].tolist() for r in rows]
        conts = {"temoin": ct}
        for t in TYPES:
            conts[t] = [c[:t0s[r]] + [int(choix[t]["jeton"][r])] + c[t0s[r] + 1:] for c, r in zip(ct, rows)]
        S, sp, r1, r8, haut = rejouer(g, [enc[r] for r in rows] * len(BRAS), [x for a in BRAS for x in conts[a]])
        haut_max = max(haut_max, haut)
        m = len(rows)
        for ai, a in enumerate(BRAS):
            sl = slice(ai * m, (ai + 1) * m)
            R[a]["S"][rows], R[a]["surp"][rows], R[a]["r1"][rows], R[a]["r8"][rows] = S[sl], sp[sl], r1[sl], r8[sl]
        if b0 == 0:
            est = (time.time() - t_debut) * P / nb / 60
            print(f"[PROGRESSION] {g.label} passes rejouées : estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
    print(f"[MESURE] {g.label} passes rejouées : {time.time() - t_debut:.0f} s", flush=True)
    return R, haut_max

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}
Z = slice(ZONE[0], ZONE[1] + 1)
N_ZONE = ZONE[1] - ZONE[0] + 1


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1v.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def contraste(x, w=8):
    """x(i) − médiane de x sur les w jetons de part et d'autre (i exclu), pour chaque i de la zone."""
    out = np.full(x.shape, np.nan)
    for i in range(ZONE[0], ZONE[1] + 1):
        voisins = np.concatenate([x[:, max(0, i - w):i], x[:, i + 1:i + 1 + w]], axis=1)
        out[:, i] = x[:, i] - np.nanmedian(voisins, axis=1)
    return out


def scores(R):
    """Scores du détecteur, position inconnue : maximum sur la zone."""
    S = R["S"].astype(float)
    return {"c1": np.nanmax(R["c1"][:, Z], 1), "r1": np.nanmax(R["r1"][:, Z], 1), "r8": np.nanmax(R["r8"][:, Z], 1),
            "entropie": S[:, ZONE[0] + 1:ZONE[1] + 2].max(1),
            "saut": np.diff(S[:, ZONE[0]:ZONE[1] + 2], axis=1).max(1),
            "surprise": R["surp"][:, Z].max(1)}


for label, ids in MODELS:
    t_start = time.time()
    model = tok = g = None
    try:
        for mid in ids:
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
        tokm, t0s, choix = generer_temoin(g, enc)
        impl = set_attention(model, "eager")
        print(f"[CONFIG] {label} : passes rejouées en attention « {impl} » (poids d'attention lus)")
        R, haut = run_replay(g, enc, tokm, t0s, choix)
        P = len(enc)
        rows = np.arange(P)
        ec = np.concatenate([np.abs(R[t]["surp"][rows, t0s] - choix[t]["surp"]) for t in TYPES])
        ecart_surp = float(ec.mean())
        avant = max(float(np.abs(R[t]["S"][rows, t0s] - R["temoin"]["S"][rows, t0s]).max()) for t in TYPES)
        print(f"[VÉRIF] {label} : surprise du jeton inséré, passe rejouée contre choix : écart moyen {ecart_surp:.1e}, max {ec.max():.1e} ; "
              f"entropie de la loi du jeton t₀ identique entre textes : écart max {avant:.1e} ; masque causal : {haut:.1e}")
        if ecart_surp > 0.05 or avant > 0.05 or haut > 1e-4:
            raise RuntimeError("alignement des positions défectueux : résultats non interprétables")
        for b in BRAS:
            R[b]["c1"] = contraste(R[b]["r1"].astype(float))
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        sc = {a: scores(R[a]) for a in BRAS}
        smax_t = sc["temoin"]["surprise"]
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts ; surprise maximale des témoins sur la zone : "
              f"médiane {np.median(smax_t):.2f} nats, 90e centile {np.percentile(smax_t, 90):.2f}")
        a = {"types": {}}
        for t in TYPES:
            e = {"surprise_mediane": float(np.median(choix[t]["surp"])), "rang_median": float(np.median(choix[t]["rang"]))}
            print(f"[JETONS] {label} {t:<9} : rang médian {e['rang_median']:.0f} ; surprise médiane {e['surprise_mediane']:.2f} nats")
            for k in ("c1", "r1", "r8", "entropie", "saut", "surprise"):
                e[k] = juger_auc(sc[t][k], sc["temoin"][k], boots, ok)
                print(f"[AUC] {label} {t:<9} max de {k:<9} (position inconnue) : {e[k]['auc']:.3f} IC95 "
                      f"[{e[k]['ci'][0]:.3f} ; {e[k]['ci'][1]:.3f}] → {e[k]['verdict']}")
            e["r1_connue"] = juger_auc(R[t]["r1"][rows, t0s], R["temoin"]["r1"][rows, t0s], boots, ok)
            print(f"[AUC] {label} {t:<9} r₁ en t₀ (position connue, comme A1r) : {e['r1_connue']['auc']:.3f} IC95 "
                  f"[{e['r1_connue']['ci'][0]:.3f} ; {e['r1_connue']['ci'][1]:.3f}]")
            for k, cle in (("c1", "c1"), ("r1", "r1"), ("r8", "r8"), ("surprise", "surp")):
                pred = ZONE[0] + np.nanargmax(R[t][cle][:, Z], 1)
                e[f"loc_{k}"] = regle_loc((pred == t0s).astype(float), boots, N_ZONE, ok)
                print(f"[LOC] {label} {t:<9} localisation par {k:<9} : exacte {e[f'loc_{k}']['part']:.1%} IC95 "
                      f"[{e[f'loc_{k}']['ci'][0]:.1%} ; {e[f'loc_{k}']['ci'][1]:.1%}] (hasard {1 / N_ZONE:.1%}) → {e[f'loc_{k}']['verdict']}")
            e["delta"] = regle_delta(sc[t]["c1"], sc["temoin"]["c1"], sc[t]["surprise"], sc["temoin"]["surprise"], boots, ok)
            print(f"[V4] {label} {t:<9} AUC(max c₁) − AUC(max surprise) = {e['delta']['delta']:+.3f} IC95 "
                  f"[{e['delta']['ci'][0]:+.3f} ; {e['delta']['ci'][1]:+.3f}] → {e['delta']['verdict']}")
            a["types"][t] = e
        a["V1"] = a["types"]["rare"]["c1"]["verdict"]
        a["V2"] = a["types"]["plausible"]["c1"]["verdict"]
        a["V3"] = a["types"]["rare"]["loc_c1"]["verdict"]
        a["V4"] = {t: a["types"][t]["delta"]["verdict"] for t in TYPES}
        print(f"[V1] {label} jeton rare, max du contraste c₁ : {a['V1']}")
        print(f"[V2] {label} jeton plausible, max du contraste c₁ : {a['V2']}")
        print(f"[V3] {label} localisation du jeton rare par c₁ : {a['V3']}")
        print(f"[V4] {label} contre la surprise : " + " ; ".join(f"{t} {v}" for t, v in a["V4"].items()))
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1v.npz"), tok=tokm, t0=t0s,
                            **{f"jeton_{t}": choix[t]["jeton"] for t in TYPES}, **{f"surp_insere_{t}": choix[t]["surp"] for t in TYPES},
                            **{f"{k}_{b}": R[b][k] for b in BRAS for k in ("S", "surp", "r1", "r8")})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1v — l'attention reçue repère-t-elle un jeton inséré sans connaître sa position ?")
v1 = {m: ALL[m]["V1"] for m in ALL if isinstance(ALL[m], dict) and "V1" in ALL[m]}
for m in v1:
    print(f"[SYNTHÈSE] {m:<18} : V1 {ALL[m]['V1']:<15} ; V2 {ALL[m]['V2']:<15} ; V3 {ALL[m]['V3']:<16} ; V4 {ALL[m]['V4']}")
these = "NON TESTABLE (modèle manquant)" if len(v1) < len(MODELS) else synthese(v1, ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « l'attention reçue repère un jeton inséré sans connaître sa position » : {these} "
      f"(prédiction SOUTENUE 45 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

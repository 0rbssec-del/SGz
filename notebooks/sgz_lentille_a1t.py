# %% [markdown]
# # SGZ A1t — L'amplification tient-elle avec une autre lentille que q = 1,45 ?
#
# **Pourquoi.** Le résultat le plus solide du projet (la trajectoire amplifie un choc au lieu de le corriger, 24 verdicts
# sur 24 de A1c à A1m) a toujours été mesuré avec l'entropie de Tsallis q = 1,45, et seulement elle (vérifié dans le code
# de A1 à A1s). Une réponse de Qwen3 (local, 28 oct.) affirmait que « q > 1 » serait la cause de l'amplification ; en tant
# que propriété du modèle, c'est faux par construction (q est notre mesure, pas un réglage du modèle), mais la question de
# robustesse est légitime : l'amplification est-elle une propriété du modèle, ou un effet de la lentille choisie ?
#
# **Dispositif (celui de A1c, volet F).** Témoin et choc générés ensemble (768 prompts, T = 0,6, 96 jetons), choc fort
# (jeton de rang 1000–5000, jamais celui du témoin) au pas t₀ = 64, même graine et même tirage des rangs que A1c–A1p
# (ancrage MD5 sur A1k). Trois passes rejouées sur les mêmes séquences : témoin (suite du témoin), choc (suite libre du
# choc), rejoué (suite du témoin, jeton choqué à t₀). Pour chaque q ∈ {0,5 ; 1 (Shannon) ; 1,45 ; 2 ; 3}, entropie de
# Tsallis S_q de la loi complète à chaque pas. Deux modèles : Qwen3-1.7B-Base (celui de A1c) et SmolLM3-3B-Base.
#
# ## Critères (fixés avant toute exécution ; règle F de A1c inchangée ; bootstrap sur les prompts, B = 1000)
#
# Pour chaque q : R_libre(k) = S_q(choc) − S_q(témoin), R_rejoué(k) = S_q(rejoué) − S_q(témoin) ; D = moyenne de
# R_rejoué sur k = 2–31 (effet direct) ; I = moyenne de R_libre − R_rejoué sur k = 2–31 (effet de la trajectoire) ;
# marge m = 0,3 × D. NON TESTABLE si IC95(D) contient 0 ; ÉQUIVALENT si IC95(I) ⊂ [−m ; +m] ; sinon COMPENSATION si
# IC95(I) < 0 ; AMPLIFICATION si IC95(I) > 0 ; sinon INDÉTERMINÉ. (La règle est relative à D : elle ne dépend pas de
# l'échelle propre à chaque q.)
# - **V1 (principal), Shannon** : verdict F à q = 1.
# - **V2, toutes les lentilles** : ROBUSTE si AMPLIFICATION pour les cinq q ; sinon le détail par q.
# - **Reproduction** : à q = 1,45, AMPLIFICATION attendue (A1c, A1p).
# - **Synthèse, thèse « l'amplification est une propriété du modèle, pas de la lentille »** : SOUTENUE si V1 =
#   AMPLIFICATION pour les deux modèles ; RÉFUTÉE si V1 ∈ {ÉQUIVALENT, COMPENSATION} pour les deux ; DÉPEND DU MODÈLE si
#   l'un est AMPLIFICATION et l'autre ÉQUIVALENT ou COMPENSATION ; sinon INDÉTERMINÉE.
# - **Descriptifs** : I / D par q (comparable d'une lentille à l'autre) ; profil de R_libre et R_rejoué par tranches de k ;
#   corrélation de Spearman de I par prompt entre q = 1 et q = 1,45 (les deux lentilles voient-elles les mêmes prompts ?).
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# V1 AMPLIFICATION : 80 % par modèle. V2 ROBUSTE : 55 % (q = 0,5, très sensible à la queue de la loi, est le plus incertain).
# Synthèse SOUTENUE 70 %, RÉFUTÉE 10 %. Spearman de I entre q = 1 et 1,45 > 0,7 : 75 %.
#
# ## Limites connues
#
# - Une seule famille de mesures (Tsallis, Shannon compris) ; pas de Rényi ni de mesure hors entropie (perte, rang).
# - Effet de la trajectoire défini comme dans A1c (le bras rejoué n'est pas un oubli « pur »).
# - Une température, deux modèles.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~10 min pour Qwen3,
# ~20 min pour SmolLM3 [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[ANCRAGE]`, `[VÉRIF]`, `[F]`, `[V1]`,
# `[V2]`, `[SYNTHÈSE]`.

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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1t" if os.path.isdir("/kaggle/working") else "sgz_a1t_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6]
QS = [0.5, 1.0, 1.45, 2.0, 3.0]
Q = 1.45                                         # lentille historique (A1c–A1s)
N_PROMPTS, GEN_LEN, T1 = 768, 96, 64
RANK_LO, RANK_HI = 1000, 5000
MID = (2, 31)
MARGE_REL = 0.3
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                  # même graine que A1c–A1s
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
A1K_MD5 = {"Qwen3-1.7B-Base": {"temoin": "cdc160929e703e6ec61452b0c6a742fa", "choc1": "a53ac3b370ec796c8e05a1d5d043115e"}}
BRAS_R = ["temoin", "choc", "rejoue"]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T1 = 16, 48, 12
    RANK_LO, RANK_HI = 20, 80
    MID = (2, 12)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-aleatoire-bis", ["smoke-qwen3-random"])]
NPOS = MID[1] + 1                                # indices t₀ … t₀ + 31 des passes rejouées
assert T1 + MID[1] <= GEN_LEN - 1
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} prompts={N_PROMPTS} longueur={GEN_LEN} t₀={T1} fenêtre k={MID[0]}–{MID[1]} ; q = {QS} ; "
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
    return {"D": d, "D_ci": dci, "I": i, "I_ci": ici, "I_sur_D": i / d if d else float("nan"), "marge": m, "verdict": v}


def synthese(v1, interpretable=True):
    vs = list(v1.values())
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if vs and all(v == "AMPLIFICATION" for v in vs):
        return "SOUTENUE"
    if vs and all(v in ("ÉQUIVALENT", "COMPENSATION") for v in vs):
        return "RÉFUTÉE"
    if "AMPLIFICATION" in vs and any(v in ("ÉQUIVALENT", "COMPENSATION") for v in vs):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


# Témoins positifs : effet direct et effet de la trajectoire synthétiques par prompt, tailles connues
rng_pc = np.random.default_rng(SEED + 1)
Pp = 768
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
D_pc = 0.1 + rng_pc.normal(0, 0.15, Pp)
centre = lambda x: x - x.mean()
checks = [("amplification", regle_F(D_pc, 0.06 + rng_pc.normal(0, 0.15, Pp), bt)["verdict"], "AMPLIFICATION"),
          ("équivalence", regle_F(D_pc, centre(rng_pc.normal(0, 0.05, Pp)), bt)["verdict"], "ÉQUIVALENT"),
          ("compensation", regle_F(D_pc, -0.06 + rng_pc.normal(0, 0.15, Pp), bt)["verdict"], "COMPENSATION"),
          ("effet direct absent", regle_F(centre(rng_pc.normal(0, 0.15, Pp)), 0.06 + rng_pc.normal(0, 0.15, Pp), bt)["verdict"],
           "NON TESTABLE (effet direct absent)"),
          ("synthèse soutenue", synthese({"a": "AMPLIFICATION", "b": "AMPLIFICATION"}), "SOUTENUE"),
          ("synthèse réfutée", synthese({"a": "ÉQUIVALENT", "b": "COMPENSATION"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "AMPLIFICATION", "b": "ÉQUIVALENT"}), "DÉPEND DU MODÈLE"),
          ("synthèse indéterminée", synthese({"a": "AMPLIFICATION", "b": "INDÉTERMINÉ"}), "INDÉTERMINÉE")]
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<22} : {got:<34} (attendu {att})")
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
        S, tk = {q: np.zeros((n, nsteps), np.float32) for q in QS}, np.zeros((n, nsteps), np.int64)
        ids, attn, _ = self.left_pad(ctxs)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        past = None
        for s in range(nsteps):
            out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
            ent = entropies_from_logits(z, QS)                     # toutes les lentilles q à chaque pas
            seeds = [SEED + 100003 * prompt_of[j] + step0 + s for j in range(n)]
            score = z / T + gumbel_noise(self.V, seeds, z.device)
            score[:, self.forbid] = -float("inf")
            if self.V > self.n_tok:
                score[:, self.n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            if force is not None:
                nxt = force(s, z, nxt)
            for q in QS:
                S[q][:, s] = ent[q].cpu().numpy()
            tk[:, s] = nxt.cpu().numpy()
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
# ---------- Choc à t₀ (comme A1c–A1p) et passes rejouées, entropie pour chaque q ----------
def Gen_masque(self, score):
    """Mêmes jetons interdits que l'échantillonnage (spéciaux, fin de texte, hors vocabulaire)."""
    score[:, self.forbid] = -float("inf")
    if self.V > self.n_tok:
        score[:, self.n_tok:] = -float("inf")
    return score


Gen.masque = Gen_masque


@torch.no_grad()
def run_main(g, enc):
    """Témoin et choc empilés, comme A1c–A1p : X au rang tiré à t₀, jamais le jeton du témoin."""
    P = len(enc)
    forced_rank = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, P)
    T = T_MAIN[0]
    res = {a: {"S": {q: np.zeros((P, GEN_LEN), np.float32) for q in QS}, "tok": np.zeros((P, GEN_LEN), np.int64)}
           for a in ("temoin", "choc")}
    info = {"X": np.zeros(P, np.int64), "rang": np.zeros(P, np.int64), "surp": np.zeros(P, np.float64)}
    npb = max(1, g.budget // 2)
    n_lots = math.ceil(P / npb)
    t0 = time.time()
    for b0 in range(0, P, npb):
        tl = time.time()
        rows_p = list(range(b0, min(P, b0 + npb)))
        n = len(rows_p)

        def force(s, z, nxt, n=n, rows_p=rows_p):
            if s < T1:
                nxt[n:] = nxt[:n]                                      # appariement exact avant le choc
            elif s == T1:
                zz = g.masque(z[n:].clone())
                lp = torch.log_softmax(z[n:].double(), dim=-1)
                order = torch.argsort(zz, dim=-1, descending=True)
                for j, r in enumerate(rows_p):
                    rk = int(min(forced_rank[r], order.shape[1] - 2))
                    if int(order[j, rk]) == int(nxt[j]):             # jamais le jeton du témoin
                        rk += 1
                    x = int(order[j, rk])
                    nxt[n + j] = x
                    info["X"][r], info["rang"][r], info["surp"][r] = x, rk, -float(lp[j, x])
            return nxt

        S, tk = g.sample([enc[r] for r in rows_p] * 2, rows_p * 2, T, GEN_LEN, 0, force)
        for ai, a in enumerate(("temoin", "choc")):
            sl = slice(ai * n, (ai + 1) * n)
            res[a]["tok"][rows_p] = tk[sl]
            for q in QS:
                res[a]["S"][q][rows_p] = S[q][sl]
        g.progress(time.time() - tl, n_lots)
    ref, fo = res["temoin"]["tok"], res["choc"]["tok"]
    same = np.array_equal(ref[:, :T1], fo[:, :T1])
    chg = float(np.mean(ref[:, T1] != fo[:, T1]))
    print(f"[VÉRIF] {g.label} : identique avant t₀ : {same} ; jeton changé à t₀ : {chg:.0%} ; rang médian du choc "
          f"{int(np.median(info['rang']))} ; surprise médiane {np.median(info['surp']):.2f} nats", flush=True)
    if not same or chg < 0.99:
        raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
    print(f"[MESURE] {g.label} génération : {time.time() - t0:.0f} s", flush=True)
    return res, info


@torch.no_grad()
def rejouer(g, ctxs, conts):
    """Entropie S_q aux indices t₀ … t₀ + 31 de la suite imposée (indice i : loi du jeton i), pour chaque q."""
    base, head = g.model.model, g.model.get_output_embeddings()
    n, Kp = len(ctxs), len(conts[0]) + 1
    ids, attn, L = g.left_pad([c + s for c, s in zip(ctxs, conts)])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    h = base(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    q0 = L - Kp + T1
    hk = h[:, q0:q0 + NPOS, :].reshape(n * NPOS, -1)
    flat = {q: np.zeros(n * NPOS, np.float64) for q in QS}
    for c0 in range(0, n * NPOS, 192):
        e = entropies_from_logits(head(hk[c0:c0 + 192].to(head.weight.device)).float(), QS)
        for q in QS:
            flat[q][c0:c0 + 192] = e[q].cpu().numpy()
    return {q: flat[q].reshape(n, NPOS).astype(np.float32) for q in QS}


@torch.no_grad()
def run_replay(g, enc, res, info):
    P = len(enc)
    SR = {a: {q: np.zeros((P, NPOS), np.float32) for q in QS} for a in BRAS_R}
    nb, lon = 12, T1 + NPOS - 1
    t0 = time.time()
    for b0 in range(0, P, nb):
        rows = list(range(b0, min(P, b0 + nb)))
        ct = [res["temoin"]["tok"][r, :lon].tolist() for r in rows]
        cc = [res["choc"]["tok"][r, :lon].tolist() for r in rows]
        conts = {"temoin": ct, "choc": cc, "rejoue": [c[:T1] + [int(info["X"][r])] + c[T1 + 1:] for c, r in zip(ct, rows)]}
        out = rejouer(g, [enc[r] for r in rows] * len(BRAS_R), [x for a in BRAS_R for x in conts[a]])
        m = len(rows)
        for ai, a in enumerate(BRAS_R):
            for q in QS:
                SR[a][q][rows] = out[q][ai * m:(ai + 1) * m]
        if b0 == 0:
            est = (time.time() - t0) * P / nb / 60
            print(f"[PROGRESSION] {g.label} passes rejouées : estimation ~{est:.0f} min", flush=True)
    print(f"[MESURE] {g.label} passes rejouées : {time.time() - t0:.0f} s", flush=True)
    return SR

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


def ecrire_resume():
    with open(os.path.join(OUT, "resume_a1t.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def rangs(x):
    return np.argsort(np.argsort(x))


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
        res, info = run_main(g, enc)
        if label in A1K_MD5 and not SMOKE_TEST:
            okt = md5(res["temoin"]["tok"][:, :96]) == A1K_MD5[label]["temoin"]
            okc = md5(res["choc"]["tok"][:, :96]) == A1K_MD5[label]["choc1"]
            print(f"[ANCRAGE] {label} : 96 premiers jetons identiques à A1k (MD5) : témoin {okt}, choc {okc}"
                  + ("" if okt and okc else " (écart numérique possible ; les contrastes, mesurés ici, restent valides)"))
        SR = run_replay(g, enc, res, info)
        for a in ("temoin", "choc"):
            d = np.abs(SR[a][Q] - res[a]["S"][Q][:, T1:T1 + NPOS])
            print(f"[VÉRIF] {label} {a} rejoué contre généré (q = {Q}) : écart moyen {d.mean():.1e}, max {d.max():.1e}", flush=True)
            if d.mean() > 0.1:
                raise RuntimeError("passe rejouée incohérente : résultats non interprétables")
        dp = max(float(np.abs(SR[a][Q][:, 0] - SR["temoin"][Q][:, 0]).max()) for a in BRAS_R)
        print(f"[VÉRIF] {label} bras identiques avant le choc (indice t₀) : écart max {dp:.1e}")
        if dp > 0.05:
            raise RuntimeError("bras désalignés avant le choc : résultats non interprétables")
        P = len(enc)
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        w = slice(MID[0], MID[1] + 1)
        a = {"F": {}, "profil": {}}
        Iq = {}
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN[0]}) : {P} prompts")
        for q in QS:
            Rl = (SR["choc"][q] - SR["temoin"][q]).astype(float)
            Rr = (SR["rejoue"][q] - SR["temoin"][q]).astype(float)
            Dp, Ip = Rr[:, w].mean(1), (Rl - Rr)[:, w].mean(1)
            Iq[q] = Ip
            f = regle_F(Dp, Ip, boots, ok)
            a["F"][str(q)] = f
            a["profil"][str(q)] = {"libre": [float(Rl[:, lo:hi + 1].mean()) for lo, hi in ((1, 1), (2, 7), (8, 15), (16, 31)) if lo <= MID[1]],
                                   "rejoue": [float(Rr[:, lo:hi + 1].mean()) for lo, hi in ((1, 1), (2, 7), (8, 15), (16, 31)) if lo <= MID[1]]}
            print(f"[F] {label} q = {q:<4} : D = {f['D']:+.4f} IC95 [{f['D_ci'][0]:+.4f} ; {f['D_ci'][1]:+.4f}] ; I = {f['I']:+.4f} "
                  f"IC95 [{f['I_ci'][0]:+.4f} ; {f['I_ci'][1]:+.4f}] ; marge ±{f['marge']:.4f} ; I/D = {f['I_sur_D']:+.2f} → {f['verdict']}")
            print(f"[PROFIL] {label} q = {q:<4} libre / rejoué par tranches de k (1 | 2–7 | 8–15 | 16–31) : "
                  + " ".join(f"{x:+.3f}/{y:+.3f}" for x, y in zip(a["profil"][str(q)]["libre"], a["profil"][str(q)]["rejoue"])))
        v1 = a["F"]["1.0"]["verdict"]
        vq = {q: a["F"][str(q)]["verdict"] for q in QS}
        v2 = "ROBUSTE" if all(v == "AMPLIFICATION" for v in vq.values()) else "NON ROBUSTE (" + ", ".join(f"q = {q} : {v}" for q, v in vq.items()) + ")"
        if not ok:
            v2 = "NON INTERPRÉTABLE"
        rho = float(np.corrcoef(rangs(Iq[1.0]), rangs(Iq[Q]))[0, 1])
        a.update({"V1": v1, "V2": v2, "spearman_I_q1_q145": rho, "surprise_mediane": float(np.median(info["surp"])),
                  "minutes": (time.time() - t_start) / 60})
        print(f"[V1] {label} Shannon (q = 1) : {v1}")
        print(f"[V2] {label} toutes les lentilles : {v2}")
        print(f"[F] {label} reproduction à q = {Q} : {vq[Q]} (attendu AMPLIFICATION, A1c et A1p) ; Spearman de I par prompt "
              f"entre q = 1 et q = {Q} : {rho:+.3f}")
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_a1t.npz"), tok_temoin=res["temoin"]["tok"], tok_choc=res["choc"]["tok"],
                            X=info["X"], rang=info["rang"], surprise=info["surp"],
                            **{f"S_{b}_q{q}": SR[b][q] for b in BRAS_R for q in QS})
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] A1t — l'amplification dépend-elle de la lentille q ?")
v1 = {m: ALL[m]["V1"] for m in ALL if isinstance(ALL[m], dict) and "V1" in ALL[m]}
for m, v in v1.items():
    print(f"[SYNTHÈSE] {m:<18} : V1 (Shannon) {v:<16} ; V2 {ALL[m]['V2']}")
these = "NON TESTABLE (modèle manquant)" if len(v1) < len(MODELS) else synthese(v1, ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « l'amplification est une propriété du modèle, pas de la lentille » : {these} "
      f"(prédiction SOUTENUE 70 %, RÉFUTÉE 10 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

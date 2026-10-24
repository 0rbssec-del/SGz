# %% [markdown]
# # SGZ A1d — L'effet direct du choc s'efface-t-il par suppression interne (attention) ou par dilution ?
#
# **Suite de A1c (24 oct. 2026).** A1c : la trajectoire AMPLIFIE le choc (T = 0,6 et 0,8) ; la forme « rétroaction par
# la trajectoire » de la thèse est RÉFUTÉE. Le retour qui existe vient de l'effacement de l'**effet direct** du jeton
# choqué (bras rejoué : +0,53 à k = 1, +0,03 à k = 32–63). Reste à savoir comment il s'efface :
# - **suppression interne** : le modèle accorde au jeton anormal moins d'attention qu'il n'en accorderait à un jeton
#   normal à la même place (ou une contribution plus faible) ;
# - **dilution** : le jeton anormal est traité comme un jeton normal ; son poids baisse seulement parce que le contexte
#   s'allonge.
#
# **Dispositif.** Mêmes prompts, même graine, même procédure de génération que A1c (768 prompts, T = 0,6 et 0,8,
# 160 jetons, choc au pas t₀ = 64) : les trajectoires doivent reproduire celles de A1c (vérifié par empreinte MD5).
# Trois bras **rejoués** (une passe avant, suite du témoin imposée, attention « eager ») qui ne diffèrent que par le
# jeton au pas t₀ : **témoin** (jeton normal), **fort** (jeton de rang 1000–5000, celui de A1c), **doux** (jeton le plus
# probable autre que celui du témoin : jeton différent mais normal, contrôle de spécificité).
#
# **Mesures**, pour la requête qui prédit le pas t₀ + k et la clé au pas t₀ :
# - **A(k)** : poids d'attention vers le jeton du pas t₀, moyenne sur les 16 têtes et les 28 couches (mesure principale) ;
# - **N(k)** : norme de la contribution du jeton t₀ à la sortie de l'attention, ‖Σ_h α_h W_O^h v_h‖, moyenne sur les
#   couches (le poids d'attention ne dit pas tout : un jeton peut être lu avec un vecteur de valeur faible).
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les prompts, B = 1000)
#
# - Fenêtre principale k = 8–31. ρ = moyenne(bras choqué) / moyenne(témoin). Marge : 20 % (une baisse d'attention de
#   moins de 20 % ne peut pas expliquer une chute de l'effet direct de ~90 % entre k = 1 et k = 8–31).
#   ÉQUIVALENT si IC95(ρ) ⊂ [0,8 ; 1,2] ; sinon SUPPRESSION si IC95(ρ) < 1 ; SAILLANCE si IC95(ρ) > 1 ;
#   sinon INDÉTERMINÉ. Même règle pour A et pour N, pour le choc fort et pour le choc doux.
# - **Verdict de la thèse, forme « suppression interne »** : SOUTENUE si, aux deux T, le choc fort donne SUPPRESSION sur
#   la même mesure (A ou N) et que le choc doux y est ÉQUIVALENT (spécificité) ; RÉFUTÉE si, aux deux T, A et N sont
#   ÉQUIVALENT ou SAILLANCE pour le choc fort ; sinon INDÉTERMINÉE. (Règle plus stricte que celle annoncée le 24 oct.,
#   qui ne portait que sur A : une suppression par les vecteurs de valeur empêche aussi la réfutation.)
# - Descriptif : profil par couche, têtes extrêmes, courbes par tranches de k, effet direct sur l'entropie (doit
#   reproduire A1c : D = +0,0773 à T = 0,6, +0,0611 à 0,8).
# - NON INTERPRÉTABLE en SMOKE_TEST ; NON TESTABLE si le témoin positif échoue ; arrêt si l'alignement échoue (jeton lu
#   à la clé ≠ jeton du pas t₀, poids d'attention absents, calcul de N incohérent, rejoué ≠ généré).
#
# **Puissance** : non calculée d'avance (aucune mesure d'attention antérieure) ; chaque valeur par prompt moyenne
# 28 × 16 têtes et 24 requêtes, sur 768 prompts.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# A, choc fort, à chaque T : SAILLANCE 40 %, ÉQUIVALENT 35 %, SUPPRESSION 15 %, INDÉTERMINÉ 10 %. N : même ordre.
# Choc doux ÉQUIVALENT : 80 %. Thèse : SOUTENUE 12 %, RÉFUTÉE 55 %, INDÉTERMINÉE 33 %.
#
# ## Limites connues
#
# - Moyenner toutes les têtes peut masquer une tête spécialisée (le profil par tête est rapporté, à titre descriptif).
# - A et N ne mesurent que la lecture directe du jeton t₀ ; l'information du choc passe aussi par les positions
#   suivantes, qui l'ont déjà absorbée.
# - Un seul modèle (Qwen3-1.7B-Base).
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : 15 à 25 min [estimation]. Budget : 2 h.
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[PROGRESSION]`, `[A]`, `[N]`, `[THÈSE]`.

# %%
import gc
import hashlib
import json
import math
import os
import re
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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1d" if os.path.isdir("/kaggle/working") else "sgz_a1d_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = [0.6, 0.8]
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 768, 160, 64
RANK_LO, RANK_HI = 1000, 5000
W_MAIN, MID_D = (8, 31), (2, 31)
WINS = [(2, 7), (8, 31), (32, 95)]
MARGE = 0.2
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261025                                   # même graine que A1c : mêmes trajectoires
MAX_HOURS = float(os.environ.get("MAX_HOURS", "2"))
ARMS_R = ["temoin", "fort", "doux"]
A1C_MD5 = {(0.6, "temoin"): "5620698c89a5f9ba0f2f6b707eabbf70", (0.6, "fort"): "9bc7d3701806a714e828bae9c90218e4",
           (0.8, "temoin"): "e8a81c2319ab2b6c71086a8a365b1c9e", (0.8, "fort"): "266ef221e6eeed5d3c5119ce9c7805dd"}
A1C_D = {0.6: 0.0773, 0.8: 0.0611}
MODELS = [("Qwen3-1.7B-Base", ["Qwen/Qwen3-1.7B-Base"])]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 16, 72, 24
    RANK_LO, RANK_HI = 20, 80
    W_MAIN, MID_D = (4, 12), (2, 12)
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
        small = sum(p.numel() for p in model.parameters()) < 3e9
        self.budget = 48 if small else 16
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
        if not SMOKE_TEST:
            ok_t, ok_f = md5(ref) == A1C_MD5[(T, "temoin")], md5(fo) == A1C_MD5[(T, "fort")]
            print(f"[VÉRIF] {g.label} T={T} : trajectoires identiques à A1c (MD5) : témoin {ok_t}, choc fort {ok_f}"
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
        self.A[l], self.N[l] = al.mean(1), N
        self.Ah[l] = al[:, :, W_MAIN[0] - 1:W_MAIN[1]].mean(-1)                # (B, H)


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
        flat[c0:c0 + 192] = entropies_from_logits(head(hk[c0:c0 + 192]).float(), [Q])[Q].cpu().numpy()
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
    out["these"] = these
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
        print(f"[CONFIG] attention pendant la génération : {getattr(model.config, '_attn_implementation', '?')}")
        res = run_main(g, enc)
        print(f"[CONFIG] attention pendant les passes rejouées : {set_attention(model, 'eager')}")
        probe = AttnProbe(model)
        res = run_replay(g, probe, enc, res)
        for hnd in probe.handles:
            hnd.remove()
        a = analyse(res, interpretable)
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label}")
        for meas, nom in (("A", "attention"), ("N", "contribution ‖αWv‖")):
            for T in T_MAIN:
                for arm in ("fort", "doux"):
                    r = a[meas][T][arm]
                    print(f"[{meas}] {label} T={T} choc {arm:<4} ({nom}, k={W_MAIN[0]}–{W_MAIN[1]}) : {r['verdict']:<12} "
                          f"ρ = {r['rho']:.3f} IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}]")
        print(f"[THÈSE] {label} (suppression interne du jeton choqué) : {a['these']}")
        for T in T_MAIN:
            for arm in ("fort", "doux"):
                r = a["D"][T][arm]
                ref_txt = f" (A1c : {A1C_D[T]:+.4f})" if arm == "fort" and not SMOKE_TEST else ""
                print(f"[D] {label} T={T} effet direct sur l'entropie, choc {arm:<4} (k={MID_D[0]}–{MID_D[1]}) : {r['D']:+.4f} "
                      f"IC95 [{r['ci'][0]:+.4f} ; {r['ci'][1]:+.4f}]{ref_txt} ; par tranches : " + " ".join(f"{v:+.3f}" for v in r["courbe"]))
        for key, c in a["courbes"].items():
            print(f"[COURBE] {label} {key:<14} par tranches de k (1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–95) : "
                  + " ".join(f"{v:.5f}" for v in c))
        for key, d in a["couches"].items():
            for arm in ("fort", "doux"):
                print(f"[COUCHES] {label} {key} ρ par couche, choc {arm:<4} : " + " ".join(f"{v:.2f}" for v in d[arm]))
        for T in T_MAIN:
            p = a["tetes"][f"{T}_part"]
            print(f"[TÊTES] {label} T={T} part des têtes avec ρ < 0,8 : fort {p['fort<0.8']:.1%}, doux {p['doux<0.8']:.1%} ; "
                  f"ρ > 1,25 : fort {p['fort>1.25']:.1%}, doux {p['doux>1.25']:.1%}")
            for h in a["tetes"][T]:
                print(f"[TÊTES] {label} T={T} couche {h['couche']:>2} tête {h['tete']:>2} : ρ fort {h['rho_fort']:.2f}, "
                      f"ρ doux {h['rho_doux']:.2f}, attention du témoin {h['attention_temoin']:.4f}")
        for p in (0, 64 if len(PROMPTS) > 64 else 0):
            T = T_MAIN[-1]
            print(f"[EXTRAIT] {label} T={T} prompt {p} jetons au pas t0 : témoin {tok.decode([int(res[('tok', T, 'temoin')][p, T0])])!r}, "
                  f"fort {tok.decode([int(res[('tok', T, 'fort')][p, T0])])!r}, doux {tok.decode([int(res[('soft', T)][p])])!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a1d.npz"), T_main=np.asarray(T_MAIN), wins=np.asarray(WINS),
                            **{f"{k}_{T}_{arm}": res[(k, T, arm)] for k in ("S", "tok") for T in T_MAIN for arm in ("temoin", "fort")},
                            **{f"soft_{T}": res[("soft", T)] for T in T_MAIN},
                            **{f"{k}_{T}_{arm}": res[(k, T, arm)].astype(np.float16 if k == "Ah" else np.float32)
                               for k in ("Srej", "A", "N", "Alw", "Nlw", "Ah") for T in T_MAIN for arm in ARMS_R})
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
    print(f"[SYNTHÈSE] {label} : thèse (suppression interne du jeton choqué) {a['these']}")
with open(os.path.join(OUT, "resume_a1d.json"), "w") as fh:
    json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "modeles": ALL}, fh,
              ensure_ascii=False, indent=1, default=str)
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

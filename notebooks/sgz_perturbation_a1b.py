# %% [markdown]
# # SGZ A1b — Retour après un choc : confirmation à puissance suffisante
#
# **Suite de A1 (23 oct. 2026).** A1 : choc efficace, effet effacé en ~8–15 jetons, mais « retour » non démontré (marge trop
# étroite, puissance non calculée) ; pas d'emballement ; indice exploratoire de retour plus lent à T = 1,1 sur 1.7B.
#
# **Calcul de puissance fait d'avance sur les données A1 (1.7B, 256 prompts).** Retour (marge = 30 % de l'effet précoce) :
# puissance ≈ 0,98 à T = 0,6 et ≈ 0,99 à T = 0,8 ; ≈ 0,14 à T = 1,0 et ≈ 0 à T = 1,1 (variabilité trop grande), d'où le
# choix des T. Ralentissement (Δ de persistance 1,1 − 0,8, k = 33–127) : demi-IC ≈ 0,058 ; puissance ≈ 0,39 / 0,77 / 0,92
# pour Δ = 0,05 / 0,08 / 0,10 ; l'effet vu en A1 (+0,066) est sans doute surestimé (sélection après coup).
#
# **Dispositif.** Qwen3-1.7B-Base (modèle de l'indice A1). 256 prompts : les 64 de A1 + 192 prompts gabarits (24 thèmes ×
# 4 débuts, en français et en anglais). 288 jetons, choc au pas t₀ = 64. Hasard commun entre bras (comme A1).
# Bras : témoin ; **choc fort** (rang 1000–5000) à T = 0,6, 0,8, 1,1 ; **choc doux** (le jeton le plus probable autre que
# celui du témoin) à T = 0,8. R(k) = S₁,₄₅(p₁, choc, t₀ + k) − S₁,₄₅(p₁, témoin, t₀ + k).
#
# ## Critères (fixés avant toute exécution ; bootstrap sur les prompts, B = 1000)
#
# - **B1, retour** (choc fort, T = 0,6 et 0,8) : E = moyenne de R sur k = 1–16 (doit avoir un IC95 > 0, sinon NON
#   TESTABLE) ; R_tard = moyenne sur k = 129–223 ; marge = 0,3 × E. RETOUR si IC95(R_tard) ⊂ [−marge ; +marge] ;
#   PAS DE RETOUR si IC95(R_tard) entièrement hors de [−marge ; +marge] ; sinon INDÉTERMINÉ.
# - **B2, emballement** (choc fort, chaque T) : f₊ − f₋ sur Δ de surprisal des 64 derniers jetons (seuil 2 nats), comme A1.
# - **RL, ralentissement** (choc fort) : persistance P(T) = moyenne de R sur k = 33–127 ; Δ = P(1,1) − P(0,8), apparié
#   par prompt. RALENTISSEMENT si IC95(Δ) > 0 ; PAS DE RALENTISSEMENT si IC95(Δ) < 0 ou ⊂ [−0,03 ; 0,03] ; sinon
#   INDÉTERMINÉ.
# - **C, sur-compensation** (choc fort, T = 0,6 et 0,8) : moyenne de R sur k = 17–64. SUR-COMPENSATION si IC95 < 0 ;
#   sinon PAS DE SUR-COMPENSATION. (Signature d'une correction active qui dépasse ; son absence n'exclut pas une
#   rétroaction très amortie.)
# - **D, taille du choc** (T = 0,8) : k½ = premier k où la courbe moyenne lissée (fenêtre 5), normalisée par sa moyenne
#   sur k = 1–3, passe sous 0,5. Si l'IC95 de l'effet précoce du choc doux contient 0 : NON TESTABLE. DÉPEND DE LA
#   TAILLE si IC95(k½ fort − k½ doux) exclut 0 ; sinon INDÉPENDANT (forme compatible avec un oubli ou une rétroaction
#   linéaire : ce test ne sépare que la correction non linéaire).
# - **Verdict de la thèse** (« rétroaction qui ramène l'incertitude sous T ≈ 1 ») : SOUTENUE si B1 = RETOUR et B2 ≠
#   EMBALLEMENT à T = 0,6 et 0,8 ; RÉFUTÉE si B1 = PAS DE RETOUR à l'une des deux ; sinon INDÉTERMINÉE. Mention « avec
#   correction active » seulement si C = SUR-COMPENSATION.
# - NON INTERPRÉTABLE en SMOKE_TEST ; NON TESTABLE si le témoin positif échoue.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# B1 = RETOUR à 0,8 : 70 % ; à 0,6 : 65 %. RL = RALENTISSEMENT : 35 % (puissance limitée, effet A1 probablement
# surestimé). C = SUR-COMPENSATION : 15 %. D = DÉPEND DE LA TAILLE : 25 %. Thèse SOUTENUE : 55 %.
#
# ## Limites connues
#
# - « Retour » = stabilité après un choc ; il ne sépare pas une correction active d'un oubli (C et D le précisent en partie).
# - Les 192 prompts gabarits sont moins variés que les 64 écrits à la main ; le log rapporte B1 sur les 64 seuls à titre
#   descriptif.
# - Un seul modèle (1.7B) : réplication sur 0.6B possible avec MODEL_IDS.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : ~40 min [estimation d'après A1]. Budget : 3 h.
# **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[PROGRESSION]`, `[B1]` `[B2]` `[RL]` `[C]` `[D]`, `[THÈSE]`.

# %%
import gc
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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1b" if os.path.isdir("/kaggle/working") else "sgz_a1b_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_GRID = [0.6, 0.8, 1.1]
ARMS = {0.6: ["temoin", "fort"], 0.8: ["temoin", "fort", "doux"], 1.1: ["temoin", "fort"]}
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 256, 288, 64
RANK_LO, RANK_HI = 1000, 5000
EARLY, MID_C, MID_RL, LATE = (1, 16), (17, 64), (33, 127), (129, 223)
MARGE_REL, RL_EQ = 0.3, 0.03
BASCULE_NATS = 2.0
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261024
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
T_RUN0 = time.time()
MODELS = [("Qwen3-1.7B-Base", ["Qwen/Qwen3-1.7B-Base"])]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 12, 72, 8
    RANK_LO, RANK_HI = 50, 200
    EARLY, MID_C, MID_RL, LATE = (1, 4), (5, 16), (9, 32), (33, 63)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_GRID} bras={ARMS} prompts={N_PROMPTS} longueur={GEN_LEN} t0={T0} modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# %%
PROMPTS = [
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
THEMES_FR = ["la photosynthèse", "la révolution industrielle", "les volcans", "le jazz", "la cuisine italienne", "les trous noirs",
             "le hockey sur glace", "les abeilles", "l'intelligence artificielle", "le Saint-Laurent", "la Renaissance",
             "les marées", "le vélo en ville", "les vaccins", "la Grèce antique", "les forêts boréales", "le théâtre",
             "l'électricité", "les échecs", "le chocolat", "les ponts suspendus", "la mémoire humaine", "les comètes",
             "le recyclage"]
DEBUTS_FR = ["Voici ce qu'il faut savoir sur {t} :", "Hier, un reportage consacré à {t} expliquait que",
             "Pour bien comprendre {t}, il faut d'abord", "L'histoire de {t} commence"]
THEMES_EN = ["photosynthesis", "the Roman Empire", "earthquakes", "blues music", "Japanese cooking", "neutron stars",
             "basketball", "ants", "machine learning", "the Mississippi River", "the Industrial Revolution", "ocean currents",
             "urban gardening", "antibiotics", "ancient Egypt", "rainforests", "opera", "magnetism", "poker", "coffee",
             "skyscrapers", "human sleep", "asteroids", "solar power"]
DEBUTS_EN = ["Here is what you need to know about {t}:", "Yesterday, a documentary about {t} explained that",
             "To really understand {t}, you first need to", "The history of {t} begins"]
GABARITS = [d.format(t=t) for t in THEMES_FR for d in DEBUTS_FR] + [d.format(t=t) for t in THEMES_EN for d in DEBUTS_EN]
PROMPTS = PROMPTS[:64] + GABARITS
assert len(PROMPTS) >= N_PROMPTS
PROMPTS = PROMPTS[:N_PROMPTS]
N_MAIN = min(64, N_PROMPTS)
print(f"[PROMPTS] {len(PROMPTS)} prompts ({N_MAIN} écrits à la main, {len(PROMPTS) - N_MAIN} gabarits)")

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
    """Bruit de Gumbel par ligne, graine explicite par ligne (mêmes nombres pour toutes les T d'un prompt)."""
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


def k_half(R):
    m = R.mean(0)
    n0 = m[:3].mean()
    if not np.isfinite(n0) or n0 <= 0:
        return float("nan")
    sm = np.convolve(m / n0, np.ones(5) / 5, mode="same")
    below = np.where(sm < 0.5)[0]
    return float(below[0] + 1) if below.size else float(len(m))


def b1(R, boots, interpretable=True):
    E, late = win(R, EARLY), win(R, LATE)
    ce = ci([E[ix].mean() for ix in boots])
    cl = ci([late[ix].mean() for ix in boots])
    marge = MARGE_REL * E.mean()
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif ce[0] <= 0:
        v = "NON TESTABLE (choc inerte)"
    elif cl[0] >= -marge and cl[1] <= marge:
        v = "RETOUR"
    elif cl[0] > marge or cl[1] < -marge:
        v = "PAS DE RETOUR"
    else:
        v = "INDÉTERMINÉ"
    return {"E": float(E.mean()), "ci_E": ce, "R_tard": float(late.mean()), "ci_tard": cl, "marge": float(marge), "verdict": v}


def b2(d, boots, interpretable=True):
    fp, fm = (d > BASCULE_NATS).astype(float), (d < -BASCULE_NATS).astype(float)
    c = ci([fp[ix].mean() - fm[ix].mean() for ix in boots])
    v = "NON INTERPRÉTABLE" if not interpretable else ("EMBALLEMENT" if c[0] > 0 else "PAS D'EMBALLEMENT DÉTECTÉ")
    return {"f_plus": float(fp.mean()), "f_moins": float(fm.mean()), "ci": c, "verdict": v}


def rl(R8, R11, boots, interpretable=True):
    d = win(R11, MID_RL) - win(R8, MID_RL)
    c = ci([d[ix].mean() for ix in boots])
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif c[0] > 0:
        v = "RALENTISSEMENT"
    elif c[1] < 0 or (c[0] >= -RL_EQ and c[1] <= RL_EQ):
        v = "PAS DE RALENTISSEMENT"
    else:
        v = "INDÉTERMINÉ"
    return {"P_0.8": float(win(R8, MID_RL).mean()), "P_1.1": float(win(R11, MID_RL).mean()), "delta": float(d.mean()),
            "ci": c, "verdict": v}


def sur_comp(R, boots, interpretable=True):
    m = win(R, MID_C)
    c = ci([m[ix].mean() for ix in boots])
    v = "NON INTERPRÉTABLE" if not interpretable else ("SUR-COMPENSATION" if c[1] < 0 else "PAS DE SUR-COMPENSATION")
    return {"R_mid": float(m.mean()), "ci": c, "verdict": v}


def taille(Rf, Rd, boots, interpretable=True):
    Ed = win(Rd, EARLY)
    ced = ci([Ed[ix].mean() for ix in boots])
    kf, kd = k_half(Rf), k_half(Rd)
    diffs = [k_half(Rf[ix]) - k_half(Rd[ix]) for ix in boots]
    c = ci(diffs)
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif ced[0] <= 0:
        v = "NON TESTABLE (choc doux inerte)"
    elif c[0] > 0 or c[1] < 0:
        v = "DÉPEND DE LA TAILLE"
    else:
        v = "INDÉPENDANT"
    return {"k_demi_fort": kf, "k_demi_doux": kd, "ci_diff": c, "E_doux": float(Ed.mean()), "ci_E_doux": ced, "verdict": v}


# Témoin positif : réponses synthétiques connues (même forme de données que les vraies)
rng_pc = np.random.default_rng(SEED + 1)
K = GEN_LEN - T0 - 1
kk = np.arange(1, K + 1)


def synth(decay, amp=1.0, persist=0.0, under=0.0, P=256, noise=0.3):
    c, w = (MID_C[0] + MID_C[1]) / 2, (MID_C[1] - MID_C[0]) / 3          # creux placé dans la fenêtre du critère C
    base = amp * decay ** kk + persist - under * np.exp(-((kk - c) / w) ** 2)
    return base[None, :] + rng_pc.normal(0, noise, (P, K))


bt = [rng_pc.integers(0, 256, 256) for _ in range(200)]
checks = [
    ("retour", b1(synth(0.8), bt)["verdict"], "RETOUR"),
    ("sans retour", b1(synth(0.8, persist=0.5), bt)["verdict"], "PAS DE RETOUR"),
    ("ralentissement", rl(synth(0.8), synth(0.97, persist=0.0), bt)["verdict"], "RALENTISSEMENT"),
    ("même vitesse", rl(synth(0.8), synth(0.8), bt)["verdict"], "PAS DE RALENTISSEMENT"),
    ("sur-compensation", sur_comp(synth(0.8, under=0.4), bt)["verdict"], "SUR-COMPENSATION"),
    ("taille : même forme", taille(synth(0.85), synth(0.85, amp=0.4), bt)["verdict"], "INDÉPENDANT"),
    ("taille : formes différentes", taille(synth(0.95), synth(0.6, amp=0.4), bt)["verdict"], "DÉPEND DE LA TAILLE"),
]
for name, got, exp in checks:
    print(f"[TÉMOIN POSITIF] {name:<27} : {got:<24} (attendu {exp})")
POSITIVE_CONTROL_OK = all(g == e for _, g, e in checks)
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON TESTABLE'}")

# %%
# ---------- Chargement ----------
class SmokeTokenizer:
    """Tokenizer jouet pour le SMOKE_TEST (caractères → identifiants)."""
    def __init__(self, V):
        self.V, self.pad_token_id, self.eos_token_id = V, 0, 1
        self.all_special_ids = [0, 1]

    def __len__(self):
        return self.V

    def encode(self, s):
        return [2 + (ord(c) % (self.V - 2)) for c in s][:12]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def transformers_major():
    return int(transformers.__version__.split(".")[0])


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == "smoke-qwen3-random":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
        torch.manual_seed(SEED)
        return Qwen3ForCausalLM(cfg).eval().to(DEVICE), SmokeTokenizer(512), "fp32 aléatoire (smoke)"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    big = any(s in model_id for s in ("7B", "8B", "9B"))
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    elif not big:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": 0}, "fp16 1 GPU"
    elif N_GPU >= 2:
        kw[dkey], kw["device_map"] = torch.float16, "auto"
        kw["max_memory"] = {i: "13GiB" for i in range(N_GPU)}
        mode = f"fp16 sur {N_GPU} GPU"
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"], mode = {"": 0}, "nf4 1 GPU"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda" and "nf4" not in mode:
        assert next(model.parameters()).dtype == torch.float16, "le modèle n'est pas en fp16"
    return model, tok, mode




# %%
# ---------- Génération appariée : témoin, choc fort, choc doux ----------
@torch.no_grad()
def run_model(model, tok, label):
    emb_dev = model.get_input_embeddings().weight.device
    V = model.get_output_embeddings().weight.shape[0]
    forbid = set(tok.all_special_ids or []) | {tok.eos_token_id}
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    forbid |= set(gen_eos if isinstance(gen_eos, (list, tuple)) else ([gen_eos] if gen_eos is not None else []))
    forbid.discard(None)
    n_tok = len(tok)
    enc = [tok.encode(p) for p in PROMPTS]
    P, Lg = len(enc), GEN_LEN
    rows_budget = 18 if sum(p.numel() for p in model.parameters()) < 3e9 else 9
    forced_rank = np.random.default_rng(SEED + 5).integers(RANK_LO, RANK_HI + 1, P)
    res = {"S": {}, "sur": {}, "tok": {}, "rang": {}}
    est_done = False
    for T in T_GRID:
        arms = ARMS[T]
        na = len(arms)
        for a in arms:
            res["S"][(T, a)] = np.zeros((P, Lg), dtype=np.float32)
            res["sur"][(T, a)] = np.zeros((P, Lg), dtype=np.float32)
            res["tok"][(T, a)] = np.zeros((P, Lg), dtype=np.int64)
            res["rang"][(T, a)] = np.zeros(P, dtype=np.int64)
        t0 = time.time()
        npb = max(1, rows_budget // na)
        for b0 in range(0, P, npb):
            rows_p = list(range(b0, min(P, b0 + npb)))
            n = len(rows_p)
            rows = rows_p * na                                   # bras empilés : [témoin…, fort…, (doux…)]
            lens = [len(enc[r]) for r in rows]
            Lp = max(lens)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.full((len(rows), Lp), pad, dtype=torch.long)
            attn = torch.zeros((len(rows), Lp), dtype=torch.long)
            for j, r in enumerate(rows):
                ids[j, Lp - lens[j]:] = torch.tensor(enc[r])
                attn[j, Lp - lens[j]:] = 1
            ids, attn = ids.to(emb_dev), attn.to(emb_dev)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            past = None
            for s in range(Lg):
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
                past = out.past_key_values
                z = out.logits[:, -1, :].float()
                if not torch.isfinite(z).all():
                    raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
                e1 = entropies_from_logits(z, [Q])[Q].cpu().numpy()
                seeds = [SEED + 100003 * r + s for r in rows]
                score = z / T + gumbel_noise(V, seeds, z.device)
                score[:, list(forbid)] = -float("inf")
                if V > n_tok:
                    score[:, n_tok:] = -float("inf")
                nxt = torch.argmax(score, dim=-1)
                if s < T0:
                    for ai in range(1, na):
                        nxt[ai * n:(ai + 1) * n] = nxt[:n]       # appariement exact avant le choc
                if s == T0:
                    for ai, a in enumerate(arms):
                        if a == "temoin":
                            continue
                        zz = z[ai * n:(ai + 1) * n].clone()
                        zz[:, list(forbid)] = -float("inf")
                        if V > n_tok:
                            zz[:, n_tok:] = -float("inf")
                        order = torch.argsort(zz, dim=-1, descending=True)
                        for j, r in enumerate(rows_p):
                            if a == "fort":
                                rk = int(min(forced_rank[r], order.shape[1] - 1))
                            else:                                  # doux : le plus probable autre que le jeton du témoin
                                rk = 0 if int(order[j, 0]) != int(nxt[j]) else 1
                            nxt[ai * n + j] = order[j, rk]
                            res["rang"][(T, a)][r] = rk
                sur = (-torch.log_softmax(z.double(), -1).gather(1, nxt[:, None])[:, 0]).cpu().numpy()
                tk = nxt.cpu().numpy()
                for ai, a in enumerate(arms):
                    sl = slice(ai * n, (ai + 1) * n)
                    res["S"][(T, a)][rows_p, s] = e1[sl]
                    res["sur"][(T, a)][rows_p, s] = sur[sl]
                    res["tok"][(T, a)][rows_p, s] = tk[sl]
                ids = nxt[:, None].to(emb_dev)
                attn = torch.cat([attn, torch.ones((len(rows), 1), dtype=attn.dtype, device=emb_dev)], dim=1)
                pos = pos[:, -1:] + 1
            del past, out
            if not est_done:
                n_lots = sum(math.ceil(P / max(1, rows_budget // len(ARMS[t]))) for t in T_GRID)
                est = (time.time() - t0) * n_lots / 60
                print(f"[PROGRESSION] {label} : premier lot en {time.time() - t0:.1f} s ; estimation ~{est:.0f} min", flush=True)
                est_done = True
                if est / 60 > MAX_HOURS:
                    raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
        ref = res["tok"][(T, "temoin")]
        for a in arms[1:]:
            same = np.array_equal(ref[:, :T0], res["tok"][(T, a)][:, :T0])
            dS = float(np.max(np.abs(res["S"][(T, "temoin")][:, :T0 + 1] - res["S"][(T, a)][:, :T0 + 1])))
            chg = float(np.mean(ref[:, T0] != res["tok"][(T, a)][:, T0]))
            print(f"[VÉRIF] {label} T={T} bras {a} : identique avant t0 : {same} ; écart d'entropie max jusqu'à t0 : {dS:.1e} ; "
                  f"jeton changé à t0 : {chg:.0%} ; rang médian {int(np.median(res['rang'][(T, a)]))}", flush=True)
            if not same or dS > 1e-2 or chg < 0.99:
                raise RuntimeError("appariement ou choc défectueux : résultats non interprétables")
        print(f"[MESURE] {label} T={T} : {time.time() - t0:.0f} s", flush=True)
    return res

# %%
# ---------- Analyse et boucle principale ----------
def R_of(res, T, a):
    return (res["S"][(T, a)][:, T0 + 1:] - res["S"][(T, "temoin")][:, T0 + 1:]).astype(float)


def d_of(res, T, a):
    last = slice(GEN_LEN - 64, GEN_LEN)
    return (res["sur"][(T, a)][:, last].mean(1) - res["sur"][(T, "temoin")][:, last].mean(1)).astype(float)


def analyse(res, interpretable):
    ok = interpretable and POSITIVE_CONTROL_OK
    P = res["S"][(T_GRID[0], "temoin")].shape[0]
    rng = np.random.default_rng(SEED + 3)
    boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]
    main = np.arange(min(64, P))
    boots_main = [rng.integers(0, len(main), len(main)) for _ in range(BOOT_N)]
    out = {"B1": {}, "B1_64": {}, "B2": {}, "C": {}}
    for T in (0.6, 0.8):
        R = R_of(res, T, "fort")
        out["B1"][T] = b1(R, boots, ok)
        out["B1_64"][T] = b1(R[main], boots_main, ok)
        out["C"][T] = sur_comp(R, boots, ok)
    for T in T_GRID:
        out["B2"][T] = b2(d_of(res, T, "fort"), boots, ok)
    out["RL"] = rl(R_of(res, 0.8, "fort"), R_of(res, 1.1, "fort"), boots, ok)
    out["D"] = taille(R_of(res, 0.8, "fort"), R_of(res, 0.8, "doux"), boots, ok)
    tranches = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 223)]
    out["courbes"] = {f"{T}|{a}": [float(R_of(res, T, a)[:, x - 1:y].mean()) for x, y in tranches if y <= GEN_LEN - T0 - 1]
                      for T in T_GRID for a in ARMS[T] if a != "temoin"}
    out["synchro_tard"] = {f"{T}|{a}": float((res["tok"][(T, a)][:, T0 + LATE[0]:] == res["tok"][(T, "temoin")][:, T0 + LATE[0]:]).mean())
                           for T in T_GRID for a in ARMS[T] if a != "temoin"}
    if interpretable and not POSITIVE_CONTROL_OK:
        for key in ("B1", "B1_64", "B2", "C"):
            for T in out[key]:
                out[key][T]["verdict"] = "NON TESTABLE (témoin positif raté)"
        out["RL"]["verdict"] = out["D"]["verdict"] = "NON TESTABLE (témoin positif raté)"
    b = out["B1"]
    if not ok:
        these = "NON INTERPRÉTABLE" if not interpretable else "NON TESTABLE"
    elif all(b[T]["verdict"] == "RETOUR" and out["B2"][T]["verdict"] != "EMBALLEMENT" for T in (0.6, 0.8)):
        these = "SOUTENUE" + (" (avec correction active)" if any(out["C"][T]["verdict"] == "SUR-COMPENSATION" for T in (0.6, 0.8)) else "")
    elif any(b[T]["verdict"] == "PAS DE RETOUR" for T in (0.6, 0.8)):
        these = "RÉFUTÉE"
    else:
        these = "INDÉTERMINÉE"
    out["these"] = these
    return out


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
        res = run_model(model, tok, label)
        a = analyse(res, interpretable)
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label}")
        for T in (0.6, 0.8):
            r = a["B1"][T]
            print(f"[B1] {label} T={T} : {r['verdict']:<26} E(k=1–16) = {r['E']:+.4f} IC95 [{r['ci_E'][0]:+.4f} ; {r['ci_E'][1]:+.4f}] ; "
                  f"R(k={LATE[0]}–{LATE[1]}) = {r['R_tard']:+.4f} IC95 [{r['ci_tard'][0]:+.4f} ; {r['ci_tard'][1]:+.4f}] ; marge ±{r['marge']:.4f}")
            r = a["B1_64"][T]
            print(f"[B1] {label} T={T} (64 prompts écrits, descriptif) : {r['verdict']} ; R tardif {r['R_tard']:+.4f} "
                  f"IC95 [{r['ci_tard'][0]:+.4f} ; {r['ci_tard'][1]:+.4f}] ; marge ±{r['marge']:.4f}")
            r = a["C"][T]
            print(f"[C] {label} T={T} : {r['verdict']} — R(k={MID_C[0]}–{MID_C[1]}) = {r['R_mid']:+.4f} IC95 [{r['ci'][0]:+.4f} ; {r['ci'][1]:+.4f}]")
        for T in T_GRID:
            r = a["B2"][T]
            print(f"[B2] {label} T={T} : {r['verdict']} — f₊ = {r['f_plus']:.3f}, f₋ = {r['f_moins']:.3f}, IC95(f₊ − f₋) "
                  f"[{r['ci'][0]:+.3f} ; {r['ci'][1]:+.3f}]")
        r = a["RL"]
        print(f"[RL] {label} : {r['verdict']} — persistance (k={MID_RL[0]}–{MID_RL[1]}) T=0,8 {r['P_0.8']:+.4f}, T=1,1 {r['P_1.1']:+.4f}, "
              f"Δ = {r['delta']:+.4f} IC95 [{r['ci'][0]:+.4f} ; {r['ci'][1]:+.4f}]")
        r = a["D"]
        print(f"[D] {label} : {r['verdict']} — k½ fort {r['k_demi_fort']:.0f}, doux {r['k_demi_doux']:.0f}, IC95(fort − doux) "
              f"[{r['ci_diff'][0]:+.1f} ; {r['ci_diff'][1]:+.1f}] ; effet précoce du choc doux {r['E_doux']:+.4f} "
              f"IC95 [{r['ci_E_doux'][0]:+.4f} ; {r['ci_E_doux'][1]:+.4f}]")
        for key, c in a["courbes"].items():
            print(f"[COURBE] {label} {key} : R par tranches de k (1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–127 | 128–223) : "
                  + " ".join(f"{v:+.3f}" for v in c) + f" ; jetons identiques en fin : {a['synchro_tard'][key]:.1%}")
        print(f"[THÈSE] {label} : {a['these']}")
        for T in (0.8,):
            for p in (0, 64 if len(PROMPTS) > 64 else 0):
                a0 = max(0, T0 - 12)
                for arm in ARMS[T]:
                    print(f"[EXTRAIT] {label} T={T} prompt {p} {arm:<6} : {tok.decode(res['tok'][(T, arm)][p, a0:T0 + 40].tolist())!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a1b.npz"), T_grid=np.asarray(T_GRID),
                            **{f"S_{T}_{arm}": res["S"][(T, arm)] for T in T_GRID for arm in ARMS[T]},
                            **{f"sur_{T}_{arm}": res["sur"][(T, arm)] for T in T_GRID for arm in ARMS[T]},
                            **{f"tok_{T}_{arm}": res["tok"][(T, arm)] for T in T_GRID for arm in ARMS[T]})
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
    print(f"[SYNTHÈSE] {label} : thèse {a['these']}")
with open(os.path.join(OUT, "resume_a1b.json"), "w") as fh:
    json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "modeles": ALL}, fh,
              ensure_ascii=False, indent=1, default=str)
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

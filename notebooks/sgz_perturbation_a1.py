# %% [markdown]
# # SGZ A1 — Réponse à une perturbation : rétroaction négative ou emballement ?
#
# **Thèse testée (Erick, 22 oct. 2026).** « La stabilisation est un mécanisme de rétroaction négative qui maintient la
# cohérence tout en permettant l'évolution. » Version testable : après un choc (un jeton improbable forcé), l'incertitude
# du modèle revient à son niveau normal sous T ≈ 1 (rétroaction négative), au lieu de diverger ou de déclencher la
# bascule observée en v3 (emballement).
#
# **Dispositif.** Pour chaque prompt et chaque T, deux générations partagent exactement les mêmes nombres aléatoires
# (Gumbel-max, graine par prompt et par pas) : elles sont identiques jusqu'au pas t₀ = 64. Au pas t₀, la génération
# « choc » reçoit un jeton forcé, tiré uniformément parmi les rangs 1000 à 5000 de la distribution du modèle (improbable
# mais pas aberrant) ; la génération « témoin » suit son cours. Ensuite, mêmes nombres aléatoires pour les deux.
# Réponse : R(k) = S₁,₄₅(p₁, choc, t₀ + k) − S₁,₄₅(p₁, témoin, t₀ + k), par prompt, k = 1 … 256.
#
# Modèles : Qwen3-0.6B-Base et Qwen3-1.7B-Base. T ∈ {0,6 ; 0,8 ; 0,9 ; 1,0 ; 1,1 ; 1,2}. 64 prompts, 320 jetons.
#
# ## Critères (fixés avant toute exécution ; bootstrap sur les prompts, B = 1000)
#
# - **Témoin d'efficacité du choc** : moyenne de R sur k = 1–16. Si son IC95 contient 0, le choc est inerte à cette T :
#   « NON TESTABLE » pour B1 à cette T.
# - **B1, retour** (par T) : R_tard = moyenne de R sur k = 129–256 ; marge = 0,1 × S moyen du témoin à cette T.
#   RETOUR si IC95(R_tard) ⊂ [−marge ; +marge] ; PAS DE RETOUR si borne basse > marge ; sinon INDÉTERMINÉ.
# - **B2, emballement** (par T) : Δ = surprisal moyenne (choc − témoin) sur les 64 derniers jetons, par prompt.
#   f₊ = part des prompts avec Δ > 2 nats ; f₋ = part avec Δ < −2 nats (niveau du hasard : le témoin bascule seul).
#   EMBALLEMENT si IC95(f₊ − f₋) > 0 ; sinon PAS D'EMBALLEMENT DÉTECTÉ.
# - **B3, temps de retour** (descriptif) : k½ = premier k où R lissé (moyenne glissante 8) passe sous la moitié de son
#   maximum sur k = 1–16.
# - **Verdict de la thèse** : SOUTENUE si, pour les 2 modèles, B1 = RETOUR et B2 ≠ EMBALLEMENT à T = 0,6, 0,8 et 0,9.
#   RÉFUTÉE si, pour les 2 modèles, B1 = PAS DE RETOUR ou B2 = EMBALLEMENT à au moins 2 de ces 3 T. Sinon INDÉTERMINÉE.
# - NON INTERPRÉTABLE en SMOKE_TEST ; NON TESTABLE si le témoin positif synthétique échoue ou si le modèle ne charge pas.
#
# ## Témoins
#
# - **Témoin positif synthétique** (avant les modèles) : réponses connues — processus qui revient (AR(1), φ = 0,8),
#   processus sans retour (choc permanent), processus qui bascule (saut de surprisal absorbant). Les règles B1 et B2 doivent
#   rendre RETOUR, PAS DE RETOUR et EMBALLEMENT.
# - **Auto-vérifications** dans le log : choc et témoin identiques avant t₀ (jetons et entropies), rang réel du jeton
#   forcé, R(k) = 0 exactement pour k ≤ 0.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# - B1 = RETOUR à T ≤ 0,9 : 55 %. B2 = EMBALLEMENT à T = 1,0–1,1 : 60 %. k½ croît avec T : 60 %.
# - Verdict de la thèse : SOUTENUE 35 %, RÉFUTÉE 25 %, INDÉTERMINÉE 40 %.
#
# ## Limites connues
#
# - Les deux bras partagent aussi le hasard APRÈS le choc (nombres aléatoires communs). C'est voulu : R mesure l'effet
#   causal du seul jeton forcé, à hasard égal. Un système contractant (stable) oublie alors le choc et les deux trajectoires
#   se resynchronisent ; un système qui s'emballe diverge malgré le hasard commun (test de « synchronisation par bruit
#   commun »). L'indice de resynchronisation des jetons est rapporté (descriptif). Limite : quand T est grande, le bruit
#   pèse plus que les logits et la resynchronisation devient plus facile ; le verdict de la thèse ne porte que sur T ≤ 0,9.
# - « Retour » au sens des systèmes dynamiques (stabilité après un choc). Un processus qui oublie simplement son passé
#   revient aussi : B1 ne distingue pas une correction active d'un simple oubli. B3 et la comparaison entre T le précisent.
# - Un seul type de choc (rangs 1000–5000) ; pas de choc fort (jeton uniforme) ni doux (rang 2).
# - Modèles de base, longueur fixe (EOS et jetons spéciaux interdits ; entropies calculées avant ce masquage).
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé**. Durée estimée : ~40 min (0.6B ~17 min, 1.7B ~22 min) [estimation].
# Budget automatique : 3 h. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[PROGRESSION]`, `[B1]` `[B2]` `[B3]` `[SYNC]`,
# `[THÈSE]`.

# %%
import gc
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_a1" if os.path.isdir("/kaggle/working") else "sgz_a1_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_GRID = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2]
Q = 1.45
N_PROMPTS, GEN_LEN, T0 = 64, 320, 64
RANK_LO, RANK_HI = 1000, 5000
EARLY, LATE = (1, 16), (129, 256)
BASCULE_NATS = 2.0
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261023
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
T_RUN0 = time.time()
MODELS = [("Qwen3-0.6B-Base", ["Qwen/Qwen3-0.6B-Base"]), ("Qwen3-1.7B-Base", ["Qwen/Qwen3-1.7B-Base"])]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_PROMPTS, GEN_LEN, T0 = 8, 48, 8
    RANK_LO, RANK_HI = 50, 200
    EARLY, LATE = (1, 4), (20, 40)
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"])]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_GRID} q={Q} prompts={N_PROMPTS} longueur={GEN_LEN} t0={T0} rangs du choc [{RANK_LO} ; {RANK_HI}] "
      f"modèles={[m[0] for m in MODELS]}")
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
assert len(PROMPTS) >= N_PROMPTS
PROMPTS = PROMPTS[:N_PROMPTS]

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
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def decide(R, sur_d, S_ctrl_mean, rng, interpretable=True):
    """R : (P, K) réponse en entropie pour k = 1..K ; sur_d : (P,) Δ de surprisal en fin de séquence."""
    P = R.shape[0]
    boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]
    e0, e1 = EARLY
    l0, l1 = LATE
    early = R[:, e0 - 1:e1].mean(1)
    late = R[:, l0 - 1:l1].mean(1)
    ce = ci([early[ix].mean() for ix in boots])
    cl = ci([late[ix].mean() for ix in boots])
    marge = 0.1 * S_ctrl_mean
    if ce[0] <= 0 <= ce[1]:
        b1 = "NON TESTABLE (choc inerte)"
    elif cl[0] >= -marge and cl[1] <= marge:
        b1 = "RETOUR"
    elif cl[0] > marge:
        b1 = "PAS DE RETOUR"
    else:
        b1 = "INDÉTERMINÉ"
    fp, fm = (sur_d > BASCULE_NATS).astype(float), (sur_d < -BASCULE_NATS).astype(float)
    cf = ci([fp[ix].mean() - fm[ix].mean() for ix in boots])
    b2 = "EMBALLEMENT" if cf[0] > 0 else "PAS D'EMBALLEMENT DÉTECTÉ"
    m = R.mean(0)
    sm = np.convolve(m, np.ones(8) / 8, mode="same")
    peak = np.max(np.abs(m[e0 - 1:e1]))
    below = np.where(np.abs(sm) < peak / 2)[0]
    k_half = int(below[below >= e1][0] + 1) if np.any(below >= e1) else None
    if not interpretable:
        b1, b2 = "NON INTERPRÉTABLE", "NON INTERPRÉTABLE"
    return {"R_tot": float(early.mean()), "ci_tot": ce, "R_tard": float(late.mean()), "ci_tard": cl, "marge": float(marge),
            "B1": b1, "f_plus": float(fp.mean()), "f_moins": float(fm.mean()), "ci_f": cf, "B2": b2, "k_demi": k_half,
            "R_moyen": m.tolist()}


# Témoin positif : processus synthétiques à réponse connue
rng_pc = np.random.default_rng(SEED + 1)
K = GEN_LEN - T0 - 1


def synth(kind, P=64):
    k = np.arange(1, K + 1)
    noise = rng_pc.normal(0, 0.3, (P, K))
    if kind == "retour":          # AR(1), φ = 0,8 : le choc décroît
        R = 1.0 * 0.8 ** k[None, :] + noise
        d = rng_pc.normal(0, 0.5, P)
    elif kind == "sans_retour":   # décalage permanent
        R = 0.5 + noise
        d = rng_pc.normal(0, 0.5, P)
    else:                         # bascule absorbante déclenchée chez 60 % des prompts
        R = 0.5 + noise
        d = rng_pc.normal(0, 0.5, P) + np.where(rng_pc.random(P) < 0.6, 3.5, 0.0)
    return R, d


pc = {}
for kind, exp1, exp2 in [("retour", "RETOUR", "PAS D'EMBALLEMENT DÉTECTÉ"), ("sans_retour", "PAS DE RETOUR", "PAS D'EMBALLEMENT DÉTECTÉ"),
                         ("bascule", "PAS DE RETOUR", "EMBALLEMENT")]:
    R, d = synth(kind)
    r = decide(R, d, S_ctrl_mean=1.0, rng=np.random.default_rng(SEED + 2))
    pc[kind] = (r["B1"] == exp1) and (r["B2"] == exp2)
    print(f"[TÉMOIN POSITIF] {kind:<12} : B1 {r['B1']:<16} (attendu {exp1}) ; B2 {r['B2']:<26} (attendu {exp2}) ; k½ = {r['k_demi']}")
POSITIVE_CONTROL_OK = all(pc.values())
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
# ---------- Génération appariée : témoin et choc ----------
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
    nb_prompts = 8 if sum(p.numel() for p in model.parameters()) < 3e9 else 4      # ×2 bras par lot
    rank_rng = np.random.default_rng(SEED + 5)
    forced_rank = rank_rng.integers(RANK_LO, RANK_HI + 1, P)                     # même rang visé pour toutes les T
    res = {k: np.zeros((len(T_GRID), P, Lg)) for k in ("S_ctrl", "S_pert", "sur_ctrl", "sur_pert")}
    res["tok_ctrl"] = np.zeros((len(T_GRID), P, Lg), dtype=np.int64)
    res["tok_pert"] = np.zeros((len(T_GRID), P, Lg), dtype=np.int64)
    res["rang_force"] = np.zeros((len(T_GRID), P), dtype=np.int64)
    res["p_force"] = np.zeros((len(T_GRID), P))
    for it, T in enumerate(T_GRID):
        t0 = time.time()
        for b0 in range(0, P, nb_prompts):
            rows_p = list(range(b0, min(P, b0 + nb_prompts)))
            rows = rows_p + rows_p                                                 # bras témoin puis bras choc
            n = len(rows_p)
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
                    nxt[n:] = nxt[:n]                      # appariement exact avant le choc (indépendant des arrondis fp16)
                if s == T0:
                    zz = z[n:].clone()
                    zz[:, list(forbid)] = -float("inf")
                    if V > n_tok:
                        zz[:, n_tok:] = -float("inf")
                    order = torch.argsort(zz, dim=-1, descending=True)
                    lp = torch.log_softmax(z[n:].double(), -1)
                    for j, r in enumerate(rows_p):
                        rk = int(min(forced_rank[r], order.shape[1] - 1))
                        nxt[n + j] = order[j, rk]
                        res["rang_force"][it, r] = rk
                        res["p_force"][it, r] = float(lp[j, order[j, rk]].exp())
                sur = (-torch.log_softmax(z.double(), -1).gather(1, nxt[:, None])[:, 0]).cpu().numpy()
                tk = nxt.cpu().numpy()
                res["S_ctrl"][it, rows_p, s], res["S_pert"][it, rows_p, s] = e1[:n], e1[n:]
                res["sur_ctrl"][it, rows_p, s], res["sur_pert"][it, rows_p, s] = sur[:n], sur[n:]
                res["tok_ctrl"][it, rows_p, s], res["tok_pert"][it, rows_p, s] = tk[:n], tk[n:]
                ids = nxt[:, None].to(emb_dev)
                attn = torch.cat([attn, torch.ones((len(rows), 1), dtype=attn.dtype, device=emb_dev)], dim=1)
                pos = pos[:, -1:] + 1
            del past, out
            if b0 == 0 and it == 0:
                est = (time.time() - t0) * math.ceil(P / nb_prompts) * len(T_GRID) / 60
                print(f"[PROGRESSION] {label} : premier lot en {time.time() - t0:.1f} s ; estimation ~{est:.0f} min", flush=True)
                if est / 60 > MAX_HOURS:
                    raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
        same_tok = np.array_equal(res["tok_ctrl"][it, :, :T0], res["tok_pert"][it, :, :T0])
        same_S = float(np.max(np.abs(res["S_ctrl"][it, :, :T0 + 1] - res["S_pert"][it, :, :T0 + 1])))
        diff_t0 = float(np.mean(res["tok_ctrl"][it, :, T0] != res["tok_pert"][it, :, T0]))
        print(f"[VÉRIF] {label} T={T} : jetons identiques avant t0 : {same_tok} ; écart max d'entropie jusqu'à t0 inclus : "
              f"{same_S:.2e} ; jeton changé à t0 : {diff_t0:.0%} ; rang forcé médian {int(np.median(res['rang_force'][it]))}, "
              f"p médiane {np.median(res['p_force'][it]):.2e} ; {time.time() - t0:.0f} s", flush=True)
        if not same_tok or same_S > 1e-2:
            raise RuntimeError(f"appariement rompu avant t0 (écart {same_S:.2e}) : résultats non interprétables")
    return res

# %%
# ---------- Analyse et boucle principale ----------
def analyse(res, interpretable):
    rng = np.random.default_rng(SEED + 3)
    out = {}
    ok = interpretable and POSITIVE_CONTROL_OK
    for it, T in enumerate(T_GRID):
        R = res["S_pert"][it, :, T0 + 1:] - res["S_ctrl"][it, :, T0 + 1:]
        last = slice(GEN_LEN - 64, GEN_LEN) if GEN_LEN - 64 > T0 else slice(T0 + 1, GEN_LEN)
        d = res["sur_pert"][it, :, last].mean(1) - res["sur_ctrl"][it, :, last].mean(1)
        r = decide(R, d, float(res["S_ctrl"][it, :, T0 + 1:].mean()), rng, interpretable=ok)
        same = res["tok_pert"][it, :, T0 + 1:] == res["tok_ctrl"][it, :, T0 + 1:]
        l0, l1 = LATE
        r["synchro_tot"] = float(same[:, EARLY[0] - 1:EARLY[1]].mean())
        r["synchro_tard"] = float(same[:, l0 - 1:l1].mean())
        if interpretable and not POSITIVE_CONTROL_OK:
            r["B1"] = r["B2"] = "NON TESTABLE (témoin positif raté)"
        out[T] = r
    return out


def thesis_verdict(all_res):
    lowT = [T for T in T_GRID if T <= 0.9]
    keys = [l for l, a in all_res.items() if "par_T" in a]
    if len(keys) < 2:
        return "NON TESTABLE (moins de 2 modèles mesurés)"
    if any(not a["interpretable"] for a in all_res.values() if "par_T" in a):
        return "NON INTERPRÉTABLE"
    sup = all(all(all_res[l]["par_T"][T]["B1"] == "RETOUR" and all_res[l]["par_T"][T]["B2"] != "EMBALLEMENT" for T in lowT) for l in keys)
    ref = all(sum(all_res[l]["par_T"][T]["B1"] == "PAS DE RETOUR" or all_res[l]["par_T"][T]["B2"] == "EMBALLEMENT" for T in lowT) >= 2 for l in keys)
    return "SOUTENUE" if sup else ("RÉFUTÉE" if ref else "INDÉTERMINÉE")


interpretable = not SMOKE_TEST
ALL = {}
for label, ids in MODELS:
    t_start = time.time()
    if (t_start - T_RUN0) / 3600 > MAX_HOURS:
        print(f"[BUDGET] {label} sauté")
        continue
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
        ALL[label] = {"verdict": "NON TESTABLE (modèle non chargé)"}
        continue
    try:
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        res = run_model(model, tok, label)
        a = {"par_T": analyse(res, interpretable), "interpretable": interpretable, "minutes": (time.time() - t_start) / 60}
        ALL[label] = a
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label}")
        for T, r in a["par_T"].items():
            print(f"[B1] {label} T={T} : {r['B1']:<26} R(k=1–16) = {r['R_tot']:+.4f} IC95 [{r['ci_tot'][0]:+.4f} ; {r['ci_tot'][1]:+.4f}] ; "
                  f"R(k=129–256) = {r['R_tard']:+.4f} IC95 [{r['ci_tard'][0]:+.4f} ; {r['ci_tard'][1]:+.4f}] ; marge ±{r['marge']:.4f}")
            print(f"[B2] {label} T={T} : {r['B2']:<26} f₊ = {r['f_plus']:.3f}, f₋ = {r['f_moins']:.3f}, "
                  f"IC95(f₊ − f₋) [{r['ci_f'][0]:+.3f} ; {r['ci_f'][1]:+.3f}]")
            print(f"[SYNC] {label} T={T} : jetons identiques choc/témoin, k = 1–16 : {r['synchro_tot']:.1%} ; "
                  f"k = 129–256 : {r['synchro_tard']:.1%}")
            print(f"[B3] {label} T={T} : k½ = {r['k_demi']} ; R moyen aux k = 1, 2, 4, 8, 16, 32, 64, 128 : "
                  + " ".join(f"{r['R_moyen'][k - 1]:+.3f}" for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= len(r['R_moyen'])))
        for T in (0.9, 1.1):
            if T in T_GRID:
                it = T_GRID.index(T)
                for p in (0, len(PROMPTS) // 2):
                    a0 = max(0, T0 - 12)
                    print(f"[EXTRAIT] {label} T={T} prompt {p} témoin : {tok.decode(res['tok_ctrl'][it, p, a0:T0 + 50].tolist())!r}")
                    print(f"[EXTRAIT] {label} T={T} prompt {p} choc   : {tok.decode(res['tok_pert'][it, p, a0:T0 + 50].tolist())!r}")
        np.savez_compressed(os.path.join(OUT, f"{label}_a1.npz"), T_grid=np.asarray(T_GRID),
                            **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in res.items()})
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min")
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}")
        ALL[label] = {"verdict": f"NON TESTABLE ({type(e).__name__})"}
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"\n{'=' * 100}")
for label, a in ALL.items():
    if "par_T" in a:
        print(f"[SYNTHÈSE] {label:<18} " + " | ".join(f"T={T}: {r['B1']} / {r['B2']}" for T, r in a["par_T"].items()))
    else:
        print(f"[SYNTHÈSE] {label:<18} {a['verdict']}")
print(f"[THÈSE] Rétroaction négative sous T ≈ 1 : {thesis_verdict(ALL)}")
with open(os.path.join(OUT, "resume_a1.json"), "w") as fh:
    json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, "these": thesis_verdict(ALL),
               "modeles": ALL}, fh, ensure_ascii=False, indent=1, default=str)
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

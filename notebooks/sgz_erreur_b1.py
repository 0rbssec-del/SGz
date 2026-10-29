# %% [markdown]
# # SGZ B1 — Le modèle réagit-il à sa propre erreur, au-delà de la surprise ? (branche 1)
#
# **Question.** A1c–A1w montrent qu'un jeton anormal fait monter l'entropie des jetons suivants, et qu'en génération libre la
# trajectoire amplifie cet effet. Branche 1 : cette réaction pourrait-elle signaler les erreurs que le modèle fait lui-même ?
# Notre propre résultat va contre : en A1w, aux rangs 1 à 5 (ceux des erreurs naturelles), la règle F donne COMPENSATION, pas
# AMPLIFICATION. Une réaction aux erreurs ne peut donc venir que de l'erreur elle-même, au-delà de sa surprise : c'est ce que
# B1 teste.
#
# **Antériorité (lue le 29 oct.).** Sun, Stolfo et Sachan (2025, arXiv 2507.12379) : des sondes sur les états cachés au signe
# « = » détectent les erreurs d'addition à plus de 90 % (Gemma 2 2B IT, Phi-3), et la bonne réponse reste décodable même quand
# le modèle se trompe. Thoria et al. (2026, arXiv 2606.06635) : l'incertitude par jeton sur une chaîne de raisonnement prédit
# l'échec ; dans le mode « engagé », les jetons après le point d'engagement ajoutent du bruit plutôt que du signal. Aucun des
# deux ne mesure la réaction *après* une erreur à surprise égale, par branches forcées et bruit commun.
#
# **Dispositif.** Chaînes de calcul synthétiques, en quelques exemples (pas de consigne) :
# `Calcul : 47 * 38 + 125 - 64` / `Étapes : 47 * 38 = 1786 ; 1786 + 125 = 1911 ; 1911 - 64 = 1847`.
# L'étape 1 (produit 2 × 2 ou 3 × 2 chiffres) est le lieu de l'erreur ; les étapes 2 et 3 réutilisent le résultat écrit.
# 1. **Sélection** : 2000 problèmes, 8 réponses libres à l'étape 1 par problème (T = 0,6). On garde les problèmes où le modèle a
#    donné au moins une fois la bonne réponse c et au moins une fois une mauvaise ; w = la mauvaise la plus fréquente. Les deux
#    branches sont donc des choix que le modèle a réellement faits.
# 2. **Surprise** : s(v) = −log p(« v ; ») dans la loi du modèle (T = 1), pour v = c et v = w ; Δs = s(w) − s(c).
# 3. **Branches** (suite libre de 40 jetons, même bruit de Gumbel à chaque position k dans toutes les branches) :
#    **C** = bonne réponse forcée ; **W** = erreur forcée ; **nul** = bonne réponse, autre bruit (effet attendu : 0) ;
#    **reprise** = C régénérée dans d'autres lots (doit redonner C).
#
# ## Mesures
#
# H = moyenne de l'entropie de Tsallis (q = 1,45) des lois des jetons k = 1 … 30 après « v ; », coupée au premier saut de
# ligne (fin de la chaîne). Par problème : T = H(W) − H(C) ; T(nul) = H(nul) − H(C).
#
# ## Critères (fixés avant toute exécution ; bootstrap apparié sur les problèmes, B = 1000)
#
# - **E1 (principal), réaction à l'erreur à surprise égale** : régression T = a + b·Δs ; a = effet de l'erreur à Δs = 0.
#   RÉACTION À L'ERREUR si la borne basse de l'IC95 de a > 0 ; ENGAGEMENT (moins d'incertitude après l'erreur) si la borne
#   haute < 0 ; ABSENTE si l'IC95 de a ⊂ [−0,5·b ; +0,5·b] (moins que l'effet d'une demi-nat de surprise) et b > 0
#   significatif ; sinon INDÉTERMINÉ. NON TESTABLE si moins de 10 % des problèmes de chaque côté de Δs = 0 (a serait une
#   extrapolation) ou moins de 100 problèmes retenus.
# - **E2, quand l'erreur est le choix le plus probable** (Δs < 0 : la bonne réponse est alors la plus surprenante) : moyenne de
#   T. L'ERREUR L'EMPORTE si la borne basse > 0 ; LA SURPRISE L'EMPORTE si la borne haute < 0 ; sinon INDÉTERMINÉ ; NON
#   TESTABLE si moins de 30 problèmes.
# - **Témoin négatif** : T(nul) doit être sans biais (IC95 contenant 0) ; sinon E1 et E2 NON INTERPRÉTABLES.
# - **Synthèse, thèse « le modèle réagit à sa propre erreur au-delà de la surprise »** : SOUTENUE si E1 = RÉACTION et E2 ≠ LA
#   SURPRISE L'EMPORTE pour les deux modèles ; RÉFUTÉE si E1 ∈ {ABSENTE, ENGAGEMENT} pour les deux ; DÉPEND DU MODÈLE si un
#   modèle suit le premier schéma et l'autre le second ; sinon INDÉTERMINÉE.
# - **Descriptifs** : sous-ensemble apparié |Δs| ≤ 0,5 ; part des problèmes où T > 0 (détecteur par paires) ; Spearman(Δs, T) ;
#   fenêtre commune aux deux branches ; profils par tranches de k ; après l'erreur, part des suites qui recopient w
#   (propagation) ou écrivent c (correction) ; exactitude par type de problème.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# E1 : RÉACTION 35 %, ABSENTE 30 %, ENGAGEMENT 10 %, INDÉTERMINÉ ou NON TESTABLE 25 %. E2 : L'ERREUR L'EMPORTE 25 %, LA
# SURPRISE L'EMPORTE 35 %, autre 40 %. Synthèse SOUTENUE 25 %, RÉFUTÉE 25 %. Propagation de w dans plus de 80 % des suites :
# 85 %.
#
# ## Limites connues
#
# - Un seul format (multiplication puis addition et soustraction), très contraint ; base few-shot, pas de modèle à consigne.
# - La branche C force la bonne réponse même quand le modèle la donne rarement : c'est alors elle l'événement surprenant
#   (d'où E2). La régression suppose un effet linéaire de Δs ; E2 et le sous-ensemble apparié servent de garde-fous.
# - Après W, les étapes 2 et 3 calculent sur w, pas sur c : leur difficulté diffère un peu (erreurs proches de c, sans biais
#   de direction attendu).
# - Deux modèles, une température, sélection sur 8 tirages.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~20 min pour Qwen3,
# ~40 min pour SmolLM3 [estimation ; la sélection, 16 000 tirages, domine]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[SÉLECTION]`, `[VÉRIF]`,
# `[TÉMOIN NÉGATIF]`, `[E1]`, `[E2]`, `[SYNTHÈSE]`.

# %%
import gc
import json
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
OUT = os.environ.get("OUT", "/kaggle/working/sgz_b1" if os.path.isdir("/kaggle/working") else "sgz_b1_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = 0.6
Q = 1.45
N_POOL, K_ECH, N_MAX, N_MIN, N_SOUS = 2000, 8, 600, 100, 30
N_EXEMPLES = 4                                   # exemples résolus dans le prompt
LONG_REP, LONG_SUITE, KMAX = 8, 40, 30           # jetons de la réponse libre, de la suite, fenêtre k = 1 … 30
ECART_APP = 0.5                                  # sous-ensemble apparié |Δs| ≤ 0,5 nat
MARGE_EQ = 0.5                                   # ABSENTE : |a| < 0,5·b
COUV = 0.10                                      # part minimale de chaque côté de Δs = 0
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261029
SEL = SEED + 31                                  # bruit de la sélection
FRAIS = SEED + 7919                              # bruit du bras nul
ETAPE_SUITE = 1000                               # graine du jeton k de la suite : base + 100003·problème + 1000 + k
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_POOL, K_ECH, N_MAX = int(os.environ.get("SMOKE_N", "120")), 6, 80
    N_MIN, N_SOUS = 10, 5
    N_EXEMPLES, LONG_REP, LONG_SUITE, KMAX = 2, 6, 24, 16
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
TRANCHES = [(lo, min(hi, KMAX)) for lo, hi in ((1, 5), (6, 15), (16, 30)) if lo <= KMAX]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] T={T_MAIN} problèmes={N_POOL} tirages={K_ECH} retenus ≤ {N_MAX} ; suite {LONG_SUITE} jetons, fenêtre k = 1–{KMAX} ; "
      f"modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# %%
# ---------- Problèmes ----------
def probleme(rng, kind):
    if kind == "2x1":                            # harnais local (SMOKE_TEST) seulement
        a, b = int(rng.integers(10, 100)), int(rng.integers(2, 10))
        c, d = int(rng.integers(10, 100)), int(rng.integers(1, 10))
    else:
        a = int(rng.integers(102, 1000)) if kind == "3x2" else int(rng.integers(12, 100))
        b = int(rng.integers(12, 100))
        c, d = int(rng.integers(100, 1000)), int(rng.integers(10, 100))
    return a, b, c, d


def chaine(a, b, c, d):
    p = a * b
    return f"Calcul : {a} * {b} + {c} - {d}\nÉtapes : {a} * {b} = {p} ; {p} + {c} = {p + c} ; {p + c} - {d} = {p + c - d}\n"


KINDS = ["2x1"] if SMOKE_TEST else ["2x2", "3x2"]
rng_pb = np.random.default_rng(SEED + 2)
EXEMPLES = "\n".join(chaine(*probleme(rng_pb, KINDS[i % len(KINDS)])) for i in range(N_EXEMPLES))
POOL = [probleme(rng_pb, KINDS[i % len(KINDS)]) for i in range(N_POOL)]
TYPE = np.array([KINDS[i % len(KINDS)] for i in range(N_POOL)])
BONNE = np.array([a * b for a, b, _, _ in POOL])
PROMPTS = [EXEMPLES + "\n" + f"Calcul : {a} * {b} + {c} - {d}\nÉtapes : {a} * {b} = " for a, b, c, d in POOL]
print(f"[CONFIG] exemple de prompt :\n{PROMPTS[0]}…")

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


def regle_e1(T, ds, boots, interpretable=True):
    """T = a + b·Δs ; a = effet de l'erreur à surprise égale."""
    n = len(T)
    if n < N_MIN:
        return {"n": n, "verdict": "NON TESTABLE (trop peu de problèmes)"}
    a, b = ols(ds, T)
    bs = np.array([ols(ds[ix], T[ix]) for ix in boots])
    aci, bci = ci(bs[:, 0]), ci(bs[:, 1])
    couv = (float(np.mean(ds < 0)), float(np.mean(ds > 0)))
    surp = bci[0] > 0
    m = MARGE_EQ * b
    if not interpretable:
        v = "NON INTERPRÉTABLE"
    elif min(couv) < COUV:
        v = "NON TESTABLE (Δs ne couvre pas 0 : extrapolation)"
    elif aci[0] > 0:
        v = "RÉACTION À L'ERREUR"
    elif aci[1] < 0:
        v = "ENGAGEMENT (moins d'incertitude après l'erreur)"
    elif surp and -m <= aci[0] and aci[1] <= m:
        v = "ABSENTE"
    else:
        v = "INDÉTERMINÉ"
    return {"n": n, "a": float(a), "a_ci": aci, "b": float(b), "b_ci": bci, "couverture": couv, "verdict": v}


def moyenne_ci(x, rng):
    return float(x.mean()), ci([x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT_N)])


def regle_e2(T, ds, rng, interpretable=True):
    """Problèmes où l'erreur est le choix le plus probable (Δs < 0) : moyenne de T."""
    x = T[ds < 0]
    if len(x) < N_SOUS:
        return {"n": int(len(x)), "verdict": "NON TESTABLE (trop peu de problèmes où l'erreur est la plus probable)"}
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


# Témoins positifs : effets synthétiques par problème, de taille connue
rng_pc = np.random.default_rng(SEED + 1)
Pp = 500
bt = [rng_pc.integers(0, Pp, Pp) for _ in range(200)]
ds_pc = rng_pc.normal(0.5, 1.5, Pp)
ds_pos = rng_pc.normal(2.0, 0.5, Pp)
bruit = lambda: rng_pc.normal(0, 0.05, Pp)
_NMIN, N_MIN = N_MIN, 100
_NSOUS, N_SOUS = N_SOUS, 30
checks = [("E1 réaction", regle_e1(0.03 + 0.02 * ds_pc + bruit(), ds_pc, bt)["verdict"], "RÉACTION À L'ERREUR"),
          ("E1 absente", regle_e1(0.04 * ds_pc + bruit(), ds_pc, bt)["verdict"], "ABSENTE"),
          ("E1 engagement", regle_e1(-0.03 + 0.02 * ds_pc + bruit(), ds_pc, bt)["verdict"],
           "ENGAGEMENT (moins d'incertitude après l'erreur)"),
          ("E1 extrapolation", regle_e1(0.03 + 0.02 * ds_pos + bruit(), ds_pos, bt)["verdict"],
           "NON TESTABLE (Δs ne couvre pas 0 : extrapolation)"),
          ("E1 trop peu", regle_e1(np.zeros(50), np.zeros(50), bt)["verdict"], "NON TESTABLE (trop peu de problèmes)"),
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
    print(f"[TÉMOIN POSITIF] {nom:<26} : {got:<50} (attendu {att})")
POSITIVE_CONTROL_OK = all(g == e for _, g, e in checks)
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON INTERPRÉTABLE'}")

# %%
# ---------- Chargement ----------
class SmokeTokenizer:
    """Tokenizer jouet par caractères, aller-retour exact (SMOKE_TEST)."""
    ALPHA = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-«»\n+*="
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
    if model_id == "smoke-qwen3-texte":
        # Qwen3 minuscule entraîné par caractères sur des chaînes de calcul : de vraies erreurs et de vraies bonnes réponses
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512)
        torch.manual_seed(SEED)
        m, tk = Qwen3ForCausalLM(cfg).to(DEVICE), SmokeTokenizer(512)
        rng = np.random.default_rng(SEED + 99)
        corpus = np.array(tk.encode("\n".join(chaine(*probleme(rng, "2x1")) for _ in range(4000))))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for _ in range(int(os.environ.get("SMOKE_PAS", "1500"))):          # fenêtres de 192 caractères : tout le prompt
            d = rng.integers(0, len(corpus) - 193, 32)
            x = torch.tensor(np.stack([corpus[i:i + 192] for i in d]), device=DEVICE)
            loss = m(input_ids=x, labels=x).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m.eval(), tk, f"fp32 entraîné sur des chaînes de calcul (smoke, perte finale {loss.item():.2f})"
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
    """K réponses libres à l'étape 1 par problème (bruit SEL, graine propre à chaque tirage). Valeur lue = les chiffres suivis
    de « ; » (None si illisible)."""
    P = len(enc)
    vals = [[None] * K for _ in range(P)]
    jobs = [(p, j) for p in range(P) for j in range(K)]
    nb = (len(jobs) + g.budget - 1) // g.budget
    for b, b0 in enumerate(range(0, len(jobs), g.budget)):
        t0 = time.time()
        lot = jobs[b0:b0 + g.budget]
        _, tk = g.sample([enc[p] for p, _ in lot], [p * 64 + j for p, j in lot], T_MAIN, LONG_REP, 0, base=SEL)
        for (p, j), row in zip(lot, tk):
            m = re.match(r"(\d+) ;", g.tok.decode(row.tolist()))
            vals[p][j] = int(m.group(1)) if m else None
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


def premier_nombre(g, row):
    m = re.match(r" (\d+) \+", g.tok.decode(row.tolist()))
    return int(m.group(1)) if m else None

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


class TropPeu(Exception):
    pass


def ecrire_resume():
    with open(os.path.join(OUT, "resume_b1.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def fmt(x, c, f="+.4f"):
    return f"{x:{f}} IC95 [{c[0]:{f}} ; {c[1]:{f}}]"


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
        juste = np.array([[v == BONNE[i] for v in r] for i, r in enumerate(vals)])
        a["selection"] = {"lisible": float(lisible.mean()), "exactitude": float(juste.mean()),
                          "exactitude_par_type": {k: float(juste[TYPE == k].mean()) for k in KINDS}}
        sel, W_val = [], []
        for i, r in enumerate(vals):
            faux = [v for v in r if v is not None and v != BONNE[i]]
            if BONNE[i] in r and faux:
                cnt = {}
                for v in faux:
                    cnt[v] = cnt.get(v, 0) + 1
                sel.append(i)
                W_val.append(max(cnt, key=lambda v: (cnt[v], -faux.index(v))))
        print(f"[SÉLECTION] {label} : réponses lisibles {lisible.mean():.1%}, exactes {juste.mean():.1%} ("
              + ", ".join(f"{k} {juste[TYPE == k].mean():.1%}" for k in KINDS) + f") ; problèmes avec au moins une bonne et une "
              f"mauvaise réponse : {len(sel)} sur {N_POOL} ; {time.time() - t1:.0f} s", flush=True)
        idx, FC, FW, Wv = [], [], [], []
        n_tok_ko = 0
        for i, w in zip(sel, W_val):
            fc = tok.encode(PROMPTS[i] + f"{BONNE[i]} ;")
            fw = tok.encode(PROMPTS[i] + f"{w} ;")
            if fc[:len(enc[i])] != enc[i] or fw[:len(enc[i])] != enc[i]:
                n_tok_ko += 1
                continue
            idx.append(i)
            FC.append(fc[len(enc[i]):])
            FW.append(fw[len(enc[i]):])
            Wv.append(w)
            if len(idx) >= N_MAX:
                break
        idx, Wv = np.array(idx, np.int64), np.array(Wv, np.int64)
        P = len(idx)
        a["selection"].update({"retenus": P, "candidats": len(sel), "jetons_incoherents": n_tok_ko})
        print(f"[SÉLECTION] {label} : {P} problèmes retenus (au plus {N_MAX}) ; découpage en jetons incohérent : {n_tok_ko}")
        if P < N_MIN:
            a["E1"] = {"n": P, "verdict": "NON TESTABLE (trop peu de problèmes)"}
            a["E2"] = {"n": 0, "verdict": "NON TESTABLE (trop peu de problèmes)"}
            ALL[label] = a
            np.savez_compressed(os.path.join(OUT, f"{label}_b1.npz"), vals=np.array([[(-1 if v is None else v) for v in r] for r in vals]))
            print(f"[E1] {label} : {a['E1']['verdict']}")
            raise TropPeu
        if n_tok_ko > 0.1 * max(1, len(sel[:N_MAX])):
            raise RuntimeError("le découpage en jetons de « v ; » ne prolonge pas celui du prompt : réponses forcées non fiables")

        # 2. Surprise des deux réponses
        s_c = surprise(g, [enc[i] for i in idx], FC)
        s_w = surprise(g, [enc[i] for i in idx], FW)
        ds = s_w - s_c

        # 3. Branches (C et W entrelacées dans les mêmes lots ; reprise : C seule, autres voisins de lot)
        t3 = time.time()
        S2, TK2 = suites(g, enc, np.repeat(idx, 2), [f for pr in zip(FC, FW) for f in pr], SEED)
        S_C, S_W, TK_C, TK_W = S2[0::2], S2[1::2], TK2[0::2], TK2[1::2]
        S_R, TK_R = suites(g, enc, idx, FC, SEED)
        S_N, TK_N = suites(g, enc, idx, FC, FRAIS)
        L_C, L_W, L_N = fenetre(g, TK_C), fenetre(g, TK_W), fenetre(g, TK_N)
        same_1 = float(np.mean(TK_R[:, 0] == TK_C[:, 0]))
        same = float(np.mean([np.array_equal(TK_R[i, :L_C[i]], TK_C[i, :L_C[i]]) for i in range(P)]))
        dS = float(np.abs(S_R[:, 0] - S_C[:, 0]).max())
        print(f"[VÉRIF] {label} reprise (C régénérée dans d'autres lots) : premier jeton identique {same_1:.1%}, chaîne entière "
              f"identique {same:.1%}, écart max d'entropie au jeton 1 {dS:.1e} ; longueur médiane de la fenêtre C {np.median(L_C):.0f}, "
              f"W {np.median(L_W):.0f} ; branches : {time.time() - t3:.0f} s", flush=True)
        if same_1 < 0.9 or same < 0.2 or dS > 0.05:
            raise RuntimeError("la régénération ne redonne pas la branche C : chemin de génération défectueux")
        a["reprise"] = {"premier": same_1, "chaine": same, "ecart_S1": dS}

        # 4. Mesures et règles
        H_C, H_W, H_N = h_post(S_C, L_C), h_post(S_W, L_W), h_post(S_N, L_N)
        T, T_nul = H_W - H_C, H_N - H_C
        Lm = np.minimum(L_C, L_W)
        T_com = h_post(S_W, Lm) - h_post(S_C, Lm)
        rngB = np.random.default_rng(SEED + 3)
        boots = [rngB.integers(0, P, P) for _ in range(BOOT_N)]
        cim = lambda x: ci([x[ix].mean() for ix in boots])
        nul = regle_nul(T_nul, boots, ok)
        t_ok = ok and nul["verdict"] == "SANS BIAIS"
        e1 = regle_e1(T, ds, boots, t_ok)
        e2 = regle_e2(T, ds, np.random.default_rng(SEED + 4), t_ok)
        app = np.abs(ds) <= ECART_APP
        e3 = moyenne_ci(T[app], np.random.default_rng(SEED + 5)) if app.sum() >= N_SOUS else (float("nan"), (float("nan"),) * 2)
        e1_com = regle_e1(T_com, ds, boots, t_ok)
        rk = lambda x: np.argsort(np.argsort(x))
        rho = float(np.corrcoef(rk(ds), rk(T))[0, 1])
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} (T = {T_MAIN}) : {P} problèmes")
        print(f"[SURPRISE] {label} : s(c) médiane {np.median(s_c):.2f} nats, s(w) médiane {np.median(s_w):.2f} nats ; Δs médian "
              f"{np.median(ds):+.2f} ; erreur plus probable que la bonne réponse (Δs < 0) : {np.mean(ds < 0):.0%}")
        print(f"[MESURE] {label} : H(C) = {H_C.mean():.4f}, H(W) = {H_W.mean():.4f} ; T = H(W) − H(C) = {fmt(T.mean(), cim(T))} ; "
              f"Spearman(Δs, T) {rho:+.3f} ; T > 0 dans {np.mean(T > 0):.1%} des problèmes")
        print(f"[TÉMOIN NÉGATIF] {label} T(nul) (bonne réponse, autre bruit) = {fmt(nul['m'], nul['ci'])} → {nul['verdict']}"
              + ("" if t_ok or not ok else " : E1 et E2 NON INTERPRÉTABLES"))
        if "a" in e1:
            print(f"[E1] {label} T = a + b·Δs : a = {fmt(e1['a'], e1['a_ci'])} ; b = {fmt(e1['b'], e1['b_ci'])} par nat ; "
                  f"Δs < 0 : {e1['couverture'][0]:.0%}, Δs > 0 : {e1['couverture'][1]:.0%} → {e1['verdict']}")
        else:
            print(f"[E1] {label} : {e1['verdict']}")
        print(f"[E2] {label} erreur plus probable que la bonne réponse ({e2['n']} problèmes) : "
              + (f"T = {fmt(e2['m'], e2['ci'])} → " if "m" in e2 else "") + e2["verdict"])
        print(f"[APPARIÉ] {label} |Δs| ≤ {ECART_APP} ({int(app.sum())} problèmes) : T = {fmt(e3[0], e3[1])}")
        if "a" in e1_com:
            print(f"[FENÊTRE COMMUNE] {label} : a = {fmt(e1_com['a'], e1_com['a_ci'])} → {e1_com['verdict']}")
        pw = [premier_nombre(g, r) for r in TK_W]
        pc = [premier_nombre(g, r) for r in TK_C]
        prop = float(np.mean([x == w for x, w in zip(pw, Wv)]))
        corr = float(np.mean([x == BONNE[i] for x, i in zip(pw, idx)]))
        copie_c = float(np.mean([x == BONNE[i] for x, i in zip(pc, idx)]))
        print(f"[SUITE] {label} après l'erreur : recopie w (propagation) {prop:.1%}, écrit c (correction) {corr:.1%}, autre "
              f"{1 - prop - corr:.1%} ; après la bonne réponse : recopie c {copie_c:.1%}")
        prof = [float((S_W - S_C)[:, lo - 1:hi].mean()) for lo, hi in TRANCHES]
        print(f"[PROFIL] {label} W − C par tranches de k ({' | '.join(f'{lo}–{hi}' for lo, hi in TRANCHES)}) : "
              + "  ".join(f"{x:+.4f}" for x in prof))
        a.update({"n": P, "s_c_med": float(np.median(s_c)), "s_w_med": float(np.median(s_w)), "ds_neg": float(np.mean(ds < 0)),
                  "H_C": float(H_C.mean()), "H_W": float(H_W.mean()), "T": float(T.mean()), "T_ci": cim(T),
                  "part_T_pos": float(np.mean(T > 0)), "spearman_ds_T": rho, "temoin_negatif": nul, "E1": e1, "E2": e2,
                  "apparie": {"n": int(app.sum()), "T": e3[0], "ci": e3[1]}, "E1_fenetre_commune": e1_com,
                  "suite": {"propagation": prop, "correction": corr, "copie_c": copie_c}, "profil": prof})
        a["minutes"] = (time.time() - t_start) / 60
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_b1.npz"), idx=idx, w=Wv, s_c=s_c, s_w=s_w, S_C=S_C, S_W=S_W, S_N=S_N,
                            TK_C=TK_C, TK_W=TK_W, TK_N=TK_N, L_C=L_C, L_W=L_W, L_N=L_N,
                            vals=np.array([[(-1 if v is None else v) for v in r] for r in vals]))
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
print(f"\n{'=' * 100}\n[SYNTHÈSE] B1 — le modèle réagit-il à sa propre erreur, au-delà de la surprise ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "E1" in ALL[m]]
v = {k: {m: ALL[m][k]["verdict"] for m in labs} for k in ("E1", "E2")}
for m in labs:
    print(f"[SYNTHÈSE] {m:<22} : E1 {v['E1'][m]:<48} ; E2 {v['E2'][m]}")
these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese(v["E1"], v["E2"], ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « le modèle réagit à sa propre erreur au-delà de la surprise » : {these} "
      f"(prédiction SOUTENUE 25 %, RÉFUTÉE 25 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

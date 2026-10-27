"""Exploratoire A1h-bis (non pré-enregistré) : analyse restreinte à Omega (rare probable contre
fréquent probable, surprise appariée), avec bootstrap, pour les deux modèles.
Usage : python3 -I resultats_a1h2_explo.py <dossier qwen> <dossier smol>"""
import sys
import numpy as np

W_MAIN = (8, 31)
MARGE = 0.2
BOOT_N = 1000
SEED = 20261027


def win(X, w):
    return X[:, w[0] - 1:w[1]].mean(1)


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)


def ratio_verdict(xs, xt, boots):
    r = float(xs.mean() / xt.mean())
    c = ci([xs[ix].mean() / xt[ix].mean() for ix in boots])
    if c[0] >= 1 - MARGE and c[1] <= 1 + MARGE:
        v = "ÉQUIVALENT"
    elif c[1] < 1:
        v = "SUPPRESSION"
    elif c[0] > 1:
        v = "SAILLANCE"
    else:
        v = "INDÉTERMINÉ"
    return r, c, v


for label, d in zip(("Qwen3-1.7B-Base", "SmolLM3-3B-Base"), sys.argv[1:3]):
    om = np.load(f"{d}/omega.npy").astype(bool)
    print(f"== {label} : Omega = {om.sum()} / {len(om)} prompts")
    rng = np.random.default_rng(SEED + 21)
    idx = np.arange(om.sum())
    boots = [rng.integers(0, om.sum(), om.sum()) for _ in range(BOOT_N)]
    Nt = win(np.load(f"{d}/N_temoin.npy")[om], W_MAIN)
    At = win(np.load(f"{d}/A_temoin.npy")[om], W_MAIN)
    for arm in ("fort", "doux", "RAs", "FP"):
        Na = win(np.load(f"{d}/N_{arm}.npy")[om], W_MAIN)
        Aa = win(np.load(f"{d}/A_{arm}.npy")[om], W_MAIN)
        rN, cN, vN = ratio_verdict(Na, Nt, boots)
        rA, cA, vA = ratio_verdict(Aa, At, boots)
        print(f"  N {arm:<6} (Ω seul) : ρ = {rN:.3f} IC95 [{cN[0]:.3f} ; {cN[1]:.3f}] {vN:<12} | "
              f"A {arm:<6} : ρ = {rA:.3f} IC95 [{cA[0]:.3f} ; {cA[1]:.3f}] {vA}")
    NRAs, NFP = win(np.load(f"{d}/N_RAs.npy")[om], W_MAIN), win(np.load(f"{d}/N_FP.npy")[om], W_MAIN)
    rP, cP, vP = ratio_verdict(NRAs, NFP, boots)
    print(f"  N RAs / N FP (apparié, Ω seul) : ρ = {rP:.3f} IC95 [{cP[0]:.3f} ; {cP[1]:.3f}] {vP}")
    # vérif dilution : part des prompts hors Omega où RAs == doux (jeton identique)
    tok_RAs, tok_doux = np.load(f"{d}/tok0_RAs.npy"), np.load(f"{d}/tok0_doux.npy")
    hors = ~om
    print(f"  hors Ω : jeton RAs == jeton doux dans {np.mean(tok_RAs[hors] == tok_doux[hors]):.1%} des prompts "
          f"(dilution attendue si proche de 100 %)")

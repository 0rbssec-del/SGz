"""A1p, analyse après coup (descriptive, non pré-enregistrée) : profil des effets selon k après t₂, contrôle de la
surprise de Y, et lien entre habituation par prompt et baisse de surprise de X. IC95 bootstrap apparié (B = 1000).
Usage : python3 profil_k.py <dossier npy Qwen3-1.7B-Base> <dossier npy Qwen3-1.7B>"""
import sys
import numpy as np
from scipy.stats import spearmanr

B, SEED, T2 = 1000, 20261028, 96
TRANCHES = ((1, 1), (2, 7), (8, 15), (16, 31))


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rng = np.random.default_rng(SEED)
for d in sys.argv[1:]:
    L = lambda k: np.load(f"{d}/{k}.npy")
    v = L("valide").astype(bool)
    S = {a: L("Srej_" + a)[v].astype(float) for a in ("temoin", "choc1", "double", "seul2", "nouveau")}
    rep, nouv, seul = S["double"] - S["choc1"], S["nouveau"] - S["choc1"], S["seul2"] - S["temoin"]
    P = int(v.sum())
    boots = [rng.integers(0, P, P) for _ in range(B)]
    print(f"\n{d} : {P} prompts")
    sx, sy = L("info_surpX2_choc1")[v], L("info_surpY2_choc1")[v]
    print(f"  surprise de Y contre X déjà vu à t₂ : égalité exacte {np.mean(sx == sy):.0%} (ex-aequo fp16), "
          f"écart absolu moyen {np.mean(np.abs(sx - sy)):.3f} nats")
    for lo, hi in TRANCHES:
        a, b, c = (x[:, T2 + lo:T2 + hi + 1].mean(1) for x in (rep, nouv, seul))
        r = ci([a[ix].mean() / b[ix].mean() for ix in boots])
        print(f"  k {lo:>2}–{hi:<2} : D_rep {a.mean():+.4f}  D_nouv {b.mean():+.4f}  D_seul {c.mean():+.4f}  "
              f"D_rep/D_nouv {a.mean() / b.mean():.2f} IC95 [{r[0]:.2f} ; {r[1]:.2f}]")
    w = slice(T2 + 2, T2 + 32)
    diff = rep[:, w].mean(1) - nouv[:, w].mean(1)
    dsurp = (L("info_surpX2_temoin") - L("info_surpX2_choc1"))[v]
    print(f"  Spearman (D_rep − D_nouv, k 2–31) contre baisse de surprise de X : {spearmanr(diff, dsurp)[0]:+.3f}")
    print(f"  rang médian de X : à t₁ {np.median(L('info_rang1')[v]):.0f} ; à t₂ après le premier choc "
          f"{np.median(L('info_rangX2')[v]):.0f}")

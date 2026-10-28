"""A1r, analyse après coup (descriptive, non pré-enregistrée) : effet au pas suivant (k = 1) contre effet en aval
(k = 2–31), lien par prompt entre surplus d'attention et effet sur l'entropie, et lien par prompt entre effet et surprise.
IC95 bootstrap sur les prompts de Ω (B = 1000).
Usage : python3 apres_coup.py <dossier npy Qwen3-1.7B-Base> <dossier npy SmolLM3-3B-Base>"""
import sys
import numpy as np
from scipy.stats import spearmanr

BRAS = ("L_mod", "NL_mod", "L_fort", "NL_fort")
rng = np.random.default_rng(20261028)
for d in sys.argv[1:]:
    L = lambda k: np.load(f"{d}/{k}.npy")
    w = slice(2, 32)
    A = {b: (L("A_" + b) - L("A_temoin"))[:, w].mean(1).astype(float) for b in BRAS}
    D = {b: (L("S_" + b) - L("S_temoin"))[:, w].mean(1).astype(float) for b in BRAS}
    D1 = {b: (L("S_" + b) - L("S_temoin"))[:, 1].astype(float) for b in BRAS}
    n = len(L("omega"))
    boots = [rng.integers(0, n, n) for _ in range(1000)]
    print(f"\n{d} : {n} prompts dans Ω")
    for lv in ("mod", "fort"):
        dA, dD, dD1 = (X["NL_" + lv] - X["L_" + lv] for X in (A, D, D1))
        c1 = np.percentile([dD1[ix].mean() for ix in boots], [2.5, 97.5])
        cd = np.percentile([dD[ix].mean() for ix in boots], [2.5, 97.5])
        print(f"  niveau {lv} : non latin − latin, effet au pas suivant (k = 1) {dD1.mean():+.3f} IC95 [{c1[0]:+.3f} ; {c1[1]:+.3f}] ; "
              f"en aval (k = 2–31) {dD.mean():+.4f} IC95 [{cd[0]:+.4f} ; {cd[1]:+.4f}] ; Spearman par prompt surplus d'attention ~ "
              f"effet en aval {spearmanr(dA, dD)[0]:+.3f}")
    for b in ("L_mod", "L_fort"):
        s = L("surprise_" + b)
        print(f"  {b} : Spearman par prompt effet en aval ~ surprise {spearmanr(D[b], s)[0]:+.3f} ; surplus d'attention ~ "
              f"surprise {spearmanr(A[b], s)[0]:+.3f}")

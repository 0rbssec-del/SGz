"""A1u2 après coup (descriptif) : v₁ porte-t-il une part spécifique de l'effet direct D ? Usage : python3 -I ce_fichier.py <dossier .npy>"""
import sys, glob, os
import numpy as np
d = sys.argv[1]
L = lambda k: np.load(os.path.join(d, k + ".npy"))
P = len(L("test"))
rng = np.random.default_rng(0)
bt = [rng.integers(0, P, P) for _ in range(1000)]
ci = lambda v: np.percentile(v, [2.5, 97.5])
al = sorted({os.path.basename(f)[2:-4] for f in glob.glob(os.path.join(d, "D_alea_*.npy"))})
D0 = L("D_aucune")
for c in ("v1_moyenne", "v1_moitie", "v1_zero"):
    Dv = L(f"D_{c}")
    r = Dv.mean() / D0.mean()
    c_ = ci([Dv[ix].mean() / D0[ix].mean() for ix in bt])
    ra = [L(f"D_{a}").mean() / D0.mean() for a in al]
    print(f"[D] {c:<11} D / D(aucune) = {r:.3f} IC95 [{c_[0]:.3f} ; {c_[1]:.3f}] ; témoins aléatoires {np.min(ra):.3f} à {np.max(ra):.3f} "
          f"(médiane {np.median(ra):.3f}) ; témoins plus bas : {sum(x < r for x in ra)} sur {len(ra)}")

"""Après coup (non pré-enregistré), sur les données d'A1w : l'amplification passe-t-elle par le premier jeton tiré après
le choc ? Prompts séparés selon que le jeton t₀ + 1 de la suite libre (même bruit que le témoin) est celui du témoin ou non.
Usage : python3 -I premier_jeton_a1w.py <dossier .npy> <t0> ..."""
import sys, os
import numpy as np
T0, W = 64, slice(2, 32)
rng = np.random.default_rng(0)
ci = lambda f, *xs: np.percentile([f(*[x[ix] for x in xs]) for ix in (rng.integers(0, len(xs[0]), len(xs[0])) for _ in range(1000))], [2.5, 97.5])
for d in sys.argv[1:]:
    L = lambda k: np.load(os.path.join(d, k + ".npy"))
    tok = L("tok")
    Stem = L("S_temoin")[:, W].mean(1)
    print(f"\n=== {d}")
    for r in (1, 50, 1000, 5000, 20000):
        Sl, Sr = L(f"S_lib{r}")[:, W].mean(1), L(f"S_rej{r}")[:, W].mean(1)
        T, I = Sl - Stem, Sl - Sr
        meme = L(f"suite_lib{r}")[:, 0] == tok[:, T0 + 1]
        a, b = I[meme], I[~meme]
        dif = ci(lambda x, y: x.mean() - y.mean(), b[:min(len(a), len(b))], a[:min(len(a), len(b))]) if min(len(a), len(b)) > 20 else (np.nan, np.nan)
        print(f"rang {r:>5} : premier jeton identique au témoin {meme.mean():.0%} ; I si identique {a.mean():+.4f} (n={len(a)}) ; "
              f"I si différent {b.mean():+.4f} (n={len(b)}) ; T si identique {T[meme].mean():+.4f} ; T si différent {T[~meme].mean():+.4f}")

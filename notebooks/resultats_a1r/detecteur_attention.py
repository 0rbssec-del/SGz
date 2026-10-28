"""A1r, après coup (descriptif, non pré-enregistré) : l'attention vers la position du jeton détecte-t-elle mieux l'anomalie que
l'entropie ? Position connue (on lit l'attention reçue par la position t₀). AUC bras contre témoin, IC95 bootstrap (B = 1000).
Usage : python3 detecteur_attention.py <dossier npy Qwen3-1.7B-Base> <dossier npy SmolLM3-3B-Base>"""
import sys
import numpy as np
from scipy.stats import rankdata


def auc(p, n):
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


rng = np.random.default_rng(20261028)
for d in sys.argv[1:]:
    L = lambda k: np.load(f"{d}/{k}.npy").astype(float)
    At, St = L("A_temoin"), L("S_temoin")
    n = len(At)
    boots = [rng.integers(0, n, n) for _ in range(1000)]
    print(f"\n{d} : {n} prompts dans Ω")
    for b in ("L_mod", "NL_mod", "L_fort", "NL_fort"):
        Ab, Sb = L("A_" + b), L("S_" + b)
        for nom, p, q in (("attention, k = 1", Ab[:, 1], At[:, 1]), ("entropie, k = 1", Sb[:, 1], St[:, 1]),
                          ("attention, k = 2–31", Ab[:, 2:32].mean(1), At[:, 2:32].mean(1)),
                          ("entropie, k = 2–31", Sb[:, 2:32].mean(1), St[:, 2:32].mean(1))):
            c = np.percentile([auc(p[ix], q[ix]) for ix in boots], [2.5, 97.5])
            print(f"  {b:<8} {nom:<20} AUC {auc(p, q):.3f} IC95 [{c[0]:.3f} ; {c[1]:.3f}]")

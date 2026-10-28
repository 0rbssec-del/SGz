"""A1o essai 2, analyse après coup (descriptive, non pré-enregistrée) : gain d'exactitude locale du gabarit tour par tour,
et décomposition selon le premier jeton de la réponse brute au tour 1 (198 = saut de ligne dans le tokenizer Qwen3).
IC95 bootstrap apparié sur les 768 chaînes (B = 1000).
Usage : python3 qualite_par_tour.py <dossier npy gabarit> <dossier npy brut>"""
import sys
import numpy as np

B, SEED, SAUT = 1000, 20261028, 198
dg, db = sys.argv[1:3]


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rng = np.random.default_rng(SEED)
for T in ("0.6", "1.0"):
    g = np.load(f"{dg}/Q_local_T{T}.npy").astype(float)
    b = np.load(f"{db}/Q_local_T{T}.npy").astype(float)
    tb = np.load(f"{db}/Q_tok_T{T}.npy")
    n = len(g)
    boots = [rng.integers(0, n, n) for _ in range(B)]
    d = g - b
    print(f"\nT = {T} : exactitude locale, gabarit − brut, par tour")
    for t in range(6):
        c = ci([d[ix, t].mean() for ix in boots])
        print(f"  tour {t + 1} : gabarit {g[:, t].mean():.3f}  brut {b[:, t].mean():.3f}  Δ {d[:, t].mean():+.3f} IC95 [{c[0]:+.3f} ; {c[1]:+.3f}]")
    m = d[:, 2:].mean(1)
    c = ci([m[ix].mean() for ix in boots])
    print(f"  tours 3–6 : Δ {m.mean():+.3f} IC95 [{c[0]:+.3f} ; {c[1]:+.3f}]")
    nl = tb[:, 0, 0] == SAUT
    print(f"  réponses brutes du tour 1 qui commencent par un saut de ligne : {nl.mean():.1%} ; exactitude brute au tour 1 "
          f"dans ce cas {b[nl, 0].mean():.3f}, au tour 2 {b[nl, 1].mean():.3f} (gabarit sur les mêmes chaînes : "
          f"{g[nl, 0].mean():.3f}, {g[nl, 1].mean():.3f})")
    for nom, sel in (("avec saut de ligne", nl), ("sans saut de ligne", ~nl)):
        dd = d[sel].mean(1)
        c = ci([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(B)])
        print(f"  chaînes {nom} ({sel.sum()}) : Δ moyen sur les 6 tours {dd.mean():+.3f} IC95 [{c[0]:+.3f} ; {c[1]:+.3f}]")

"""A1k — complément APRÈS COUP (non pré-enregistré) à decomposition_covariance.py.
Le terme « moyennes » M = moyenne_l(H · ᾱ_l · Ū_l) contient une covariance entre couches. On le sépare :
    I = H · moyenne_l(ᾱ_l) · moyenne_l(Ū_l)     attention globale × valeur globale (indépendance totale)
    M / I                                        covariance entre couches (l'attention monte-t-elle dans les couches
                                                 où la valeur pèse le plus ?)
Identité : ρ_N = ρ_I × ρ_couches × ρ_têtes × ρ_cohérence.
Usage : python3 -I decomposition_couches.py <dossier des sous-dossiers .npy>
"""
import os
import sys

import numpy as np

W, B, SEED = (8, 31), 1000, 20261029


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def charge(d, arm):
    Ah = np.load(f"{d}/Ah_{arm}.npy").astype(np.float64)
    U = np.load(f"{d}/U_{arm}.npy").astype(np.float64)
    N = np.load(f"{d}/N_{arm}.npy").astype(np.float64)[:, W[0] - 1:W[1]].mean(1)
    H = Ah.shape[2]
    a, u = Ah.mean(2), U.mean(2)                                     # (P, L) moyennes sur les têtes
    return H * a.mean(1) * u.mean(1), H * (a * u).mean(1), (Ah * U).sum(2).mean(1), N


def fac(t, f, ix):
    It, Mt, St, Nt = (x[ix].mean() for x in t)
    If, Mf, Sf, Nf = (x[ix].mean() for x in f)
    return (Nf / Nt, If / It, (Mf / If) / (Mt / It), (Sf / Mf) / (St / Mt), (Nf / Sf) / (Nt / St))


rng = np.random.default_rng(SEED)
noms = ("attention × valeur globales", "covariance entre couches", "covariance entre têtes", "cohérence (alignement)")
for m in sorted(os.listdir(sys.argv[1])):
    d = os.path.join(sys.argv[1], m)
    t, f = charge(d, "temoin"), charge(d, "fort")
    P = len(t[0])
    pt = fac(t, f, np.arange(P))
    bs = np.array([fac(t, f, rng.integers(0, P, P)) for _ in range(B)])
    parts = np.log(bs[:, 1:]) / np.log(bs[:, [0]])
    print(f"\n== {m} : ρ_N = {pt[0]:.3f} = " + " · ".join(f"×{x:.3f}" for x in pt[1:]))
    for j, nom in enumerate(noms):
        c = ci(parts[:, j])
        print(f"   part {nom:<28} : {np.log(pt[j + 1]) / np.log(pt[0]):+.0%}  IC95 [{c[0]:+.0%} ; {c[1]:+.0%}]")

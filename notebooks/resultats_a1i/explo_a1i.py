"""Exploratoire A1i (non pré-enregistré).
Usage : python3 -I explo_a1i.py <dossier contenant un sous-dossier par modèle (npy des zip *_a1i.zip)>"""
import os
import sys
import numpy as np

d, T0 = sys.argv[1], 64
for m in sorted(os.listdir(d)):
    p = os.path.join(d, m)
    L = lambda n: np.load(os.path.join(p, n + ".npy"))
    print(f"== {m}")
    for T in ("0.6", "0.8"):
        St, Sf = L(f"S_{T}_temoin"), L(f"S_{T}_fort")
        Rt, Rf = L(f"Srej_{T}_temoin"), L(f"Srej_{T}_fort")
        late = (Sf - St)[:, T0 + 33:T0 + 96].mean(1)                     # bras libre, k = 32–95
        I = ((Sf - St) - (Rf - Rt))[:, T0 + 3:T0 + 32].mean(1)           # trajectoire, k = 2–31
        print(f"  T={T} S0 témoin (k=2–31) {Rt[:, T0 + 3:T0 + 32].mean():.3f} ; libre k=32–95 : moyenne {late.mean():+.3f}, "
              f"médiane {np.median(late):+.3f}, part > +0,5 {np.mean(late > 0.5):.1%} ; "
              f"I par prompt : moyenne {I.mean():+.3f}, médiane {np.median(I):+.3f}")
    At, Af = L("Alw_0.6_temoin"), L("Alw_0.6_fort")                     # (P, couches, fenêtres)
    same = [l for l in range(At.shape[1]) if np.allclose(At[:, l, 1], Af[:, l, 1], rtol=1e-3, atol=0)]
    print(f"  couches où l'attention vers t0 est identique témoin/choc (k=8–31, T=0.6) : {same}")
    if same:
        l = same[0]
        print(f"    couche {l} : attention moyenne vers t0 {At[:, l, 1].mean():.5f}, écart-type entre prompts {At[:, l, 1].std():.5f}")

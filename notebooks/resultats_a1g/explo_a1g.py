"""Exploratoire A1g (non pré-enregistré) : entropie de base du témoin et distribution de D par prompt.
Usage : python3 -I explo_a1g.py <dossier des .npy de Qwen3-4B_a1g.zip>"""
import sys
import numpy as np
d, T0 = sys.argv[1], 64
for T in ("0.6", "0.8"):
    t, f, s = (np.load(f"{d}/Srej_{T}_{a}.npy") for a in ("temoin", "fort", "doux"))
    Dp = (f - t)[:, T0 + 2:T0 + 32].mean(1)
    print(f"T={T} S témoin avant t0 {t[:, T0 - 32:T0].mean():.3f} ; k=2–31 {t[:, T0 + 2:T0 + 32].mean():.3f} ; "
          f"D fort {Dp.mean():.4f} (médiane {np.median(Dp):.4f}, part > 0 {(Dp > 0).mean():.2f}) ; "
          f"D doux {(s - t)[:, T0 + 2:T0 + 32].mean():.4f}")

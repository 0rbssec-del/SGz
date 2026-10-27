"""Critère C2 de l'étape A (commit 48dea59, fixé avant tout calcul) appliqué hors échantillon aux données A1m.
G = centre de gravité en profondeur de max(δ̄_l, 0), δ̄_l = moyenne sur les prompts de ᾱ_l(fort) − ᾱ_l(témoin)
(attention moyenne sur les têtes, fenêtre k = 8–31) ; APPUYÉE si borne basse IC95 > 0,67, CONTREDITE si borne haute < 0,5."""
import sys
import numpy as np

B, SEED = 1000, 20261029


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def centre(dbar):
    pos = np.clip(dbar, 0, None)
    z = np.linspace(0, 1, len(dbar))
    return float((pos * z).sum() / pos.sum()) if pos.sum() > 0 else np.nan


rng = np.random.default_rng(SEED)
for d in sys.argv[1:]:
    for T in ("0.6", "0.8"):
        af = np.load(f"{d}/Ah_{T}_fort.npy").astype(np.float64).mean(2)
        at = np.load(f"{d}/Ah_{T}_temoin.npy").astype(np.float64).mean(2)
        delta = af - at
        P = len(delta)
        G = centre(delta.mean(0))
        Gc = ci([centre(delta[rng.integers(0, P, P)].mean(0)) for _ in range(B)])
        v = "APPUYÉE" if Gc[0] > 0.67 else ("CONTREDITE" if Gc[1] < 0.5 else "INDÉTERMINÉE")
        top = [int(l) for l in np.argsort(delta.mean(0))[::-1][:3]]
        print(f"{d.rstrip('/').split('/')[-1]:<7} T={T} : G = {G:.2f} IC95 [{Gc[0]:.2f} ; {Gc[1]:.2f}] → C2 {v} ; "
              f"couches au plus fort surplus {top} sur 0–{delta.shape[1] - 1}")

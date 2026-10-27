"""A1k — étape A de la thèse « routage de l'anomalie vers les couches profondes » (analyse sur données existantes,
critère fixé et commité AVANT tout calcul, 27 oct. 2026).

Pour chaque prompt p et chaque couche l (fenêtre k = 8–31, déjà moyennée dans Ah) :
    ᾱ_{p,l} = moyenne sur les têtes de l'attention vers t₀ ;   δ_{p,l} = ᾱ_{p,l}(fort) − ᾱ_{p,l}(témoin)
    w_l     = Ū_l centré-réduit sur les couches, Ū_l = moyenne (prompts, têtes) de ‖W_O^h v_h(t₀)‖ du bras fort
    D_p     = Σ_l δ_{p,l} w_l / Σ_l |δ_{p,l}|      indice de déplacement vers les couches de grande sortie, ∈ [−max|w|, max|w|]
    I_p     = amplification par la trajectoire, même définition que A1k (volet F, k = 2–31, T = 0,6)
    G       = centre de gravité en profondeur (0 = 1ʳᵉ couche, 1 = dernière) de max(δ̄_l, 0), δ̄ = moyenne sur les prompts

Critères (fixés avant calcul) :
  C4 (déplacement ↔ amplification, Spearman D_p ~ I_p, IC95 bootstrap B = 1000) :
     APPUYÉE si borne basse > 0,2 dans ≥ 3 modèles sur 5 ; RÉFUTÉE si l'IC contient 0 dans ≥ 4 sur 5 ; sinon INDÉTERMINÉE.
  C2 (le surplus d'attention va aux couches tardives, G avec IC95 bootstrap) :
     APPUYÉE si borne basse de G > 0,67 dans ≥ 4 sur 5 ; CONTREDITE si borne haute < 0,5 dans ≥ 4 sur 5 ; sinon INDÉTERMINÉE.
  Contrôle : D̄ (moyenne de D_p) doit être > 0 si le déplacement « entre couches » d'A1k est retrouvé par cet indice.
Une C4 réfutée fait tomber la spéculation « routage → amplification », pas le constat de déplacement lui-même.

Usage : python3 -I routage_couches.py <dossier contenant un sous-dossier .npy par modèle>
"""
import os
import sys

import numpy as np

T0, MID_F = 64, (2, 31)
B, SEED = 1000, 20261029


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)


def rang(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(rang(x), rang(y))[0, 1])


def win(X, w):
    return X[:, w[0] - 1:w[1]].mean(1)


def centre(dbar):
    pos = np.clip(dbar, 0, None)
    z = np.linspace(0, 1, len(dbar))
    return float((pos * z).sum() / pos.sum()) if pos.sum() > 0 else np.nan


def main(racine):
    rng = np.random.default_rng(SEED)
    c4, c2 = [], []
    print("Étape A — déplacement de l'attention vers les couches de grande sortie, et lien avec l'amplification I")
    for m in sorted(os.listdir(racine)):
        d = os.path.join(racine, m)
        ld = lambda n: np.load(f"{d}/{n}.npy").astype(np.float64)
        af, at = ld("Ah_fort").mean(2), ld("Ah_temoin").mean(2)                  # (P, L)
        delta = af - at
        Ub = ld("U_fort").mean((0, 2))                                            # (L,)
        w = (Ub - Ub.mean()) / Ub.std()
        D = (delta * w).sum(1) / np.abs(delta).sum(1)
        Rlib = (ld("S_fort") - ld("S_temoin"))[:, T0 + 1:]
        Rrej = (ld("Srej_fort") - ld("Srej_temoin"))[:, T0 + 1:]
        I = win(Rlib, MID_F) - win(Rrej, MID_F)
        P = len(D)
        boots = [rng.integers(0, P, P) for _ in range(B)]
        r = spearman(D, I)
        rc = ci([spearman(D[ix], I[ix]) for ix in boots])
        G = centre(delta.mean(0))
        Gc = ci([centre(delta[ix].mean(0)) for ix in boots])
        Dc = ci([D[ix].mean() for ix in boots])
        c4.append("A" if rc[0] > 0.2 else ("R" if rc[0] <= 0 <= rc[1] else "I"))
        c2.append("A" if Gc[0] > 0.67 else ("C" if Gc[1] < 0.5 else "I"))
        top = np.argsort(delta.mean(0))[::-1][:3]
        print(f"\n== {m}  (P = {P}, L = {delta.shape[1]})")
        print(f"   D̄ = {D.mean():+.3f}  IC95 [{Dc[0]:+.3f} ; {Dc[1]:+.3f}]   (contrôle : > 0 attendu)")
        print(f"   Spearman(D_p, I_p) = {r:+.3f}  IC95 [{rc[0]:+.3f} ; {rc[1]:+.3f}]   → C4 {c4[-1]}")
        print(f"   centre de gravité G = {G:.2f}  IC95 [{Gc[0]:.2f} ; {Gc[1]:.2f}]   → C2 {c2[-1]}")
        print(f"   couches au plus fort surplus δ̄ : {[int(l) for l in top]} sur 0–{delta.shape[1] - 1}")
    n = len(c4)
    v4 = "APPUYÉE" if c4.count("A") >= 3 else ("RÉFUTÉE" if c4.count("R") >= 4 else "INDÉTERMINÉE")
    v2 = "APPUYÉE" if c2.count("A") >= 4 else ("CONTREDITE" if c2.count("C") >= 4 else "INDÉTERMINÉE")
    print(f"\nSynthèse sur {n} modèles : C4 (déplacement ↔ amplification) {v4} ; C2 (couches tardives) {v2}")


if __name__ == "__main__":
    main(sys.argv[1])

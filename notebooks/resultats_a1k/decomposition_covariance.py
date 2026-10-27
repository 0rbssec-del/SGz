"""A1k — décomposition de l'excès de contribution N en trois facteurs (analyse sur données existantes, critère fixé
avant calcul le 27 oct. ; cf. message « Si la covariance porte la majorité de l'écart, la piste tient ; si elle est
proche de zéro, elle tombe »).

Pour chaque couche l et chaque bras a (fenêtre k = 8–31), avec α_h l'attention de la tête h vers t₀ (moyenne sur k)
et U_h = ‖W_O^h v_h(t₀)‖ :
    M = H · moyenne_h(α) · moyenne_h(U)        attention moyenne × valeur moyenne (aucune covariance entre têtes)
    S = Σ_h α_h U_h = M + H · cov_h(α, U)       ajoute la covariance : l'attention va-t-elle aux têtes à grande valeur ?
    N = moyenne_k ‖Σ_h α_h(k) W_O^h v_h‖ ≤ S    ajoute l'alignement des têtes (cohérence C = N / S ≤ 1)
Moyennes sur les couches puis sur les prompts. Identité exacte :
    ρ_N = ρ_M × ρ_cov × ρ_coh,  ρ_cov = (S/M)_fort / (S/M)_témoin,  ρ_coh = (N/S)_fort / (N/S)_témoin.
Part de chaque facteur = log ρ_facteur / log ρ_N, IC95 bootstrap apparié sur les prompts (B = 1000).

Critère (fixé avant calcul) pour la piste « redistribution entre têtes » (facteur covariance) :
    TIENT si la borne basse de l'IC95 de sa part > 0,5 ;
    TOMBE si l'IC95 de sa part contient 0 ou si sa borne haute < 0,2 ;
    sinon PARTIELLE.

Usage : python3 -I decomposition_covariance.py <dossier contenant un sous-dossier .npy par modèle (zips *_a1k.zip)>
"""
import os
import sys

import numpy as np

W = (8, 31)
B, SEED = 1000, 20261028


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)


def termes(d, arm):
    Ah = np.load(f"{d}/Ah_{arm}.npy").astype(np.float64)            # (P, L, H) attention par tête, moyenne sur k = 8–31
    U = np.load(f"{d}/U_{arm}.npy").astype(np.float64)              # (P, L, H) ‖W_O^h v_h(t₀)‖
    N = np.load(f"{d}/N_{arm}.npy").astype(np.float64)              # (P, K) moyenne sur les couches
    H = Ah.shape[2]
    M = (H * Ah.mean(2) * U.mean(2)).mean(1)                        # (P,)
    S = (Ah * U).sum(2).mean(1)                                      # (P,)
    Nw = N[:, W[0] - 1:W[1]].mean(1)                                 # (P,)
    A_chk = np.load(f"{d}/A_{arm}.npy")[:, W[0] - 1:W[1]].mean(1)    # vérification : moyenne têtes et couches de Ah
    return M, S, Nw, float(np.max(np.abs(Ah.mean((1, 2)) - A_chk) / (np.abs(A_chk) + 1e-12)))


def facteurs(t, f, ix):
    Mt, St, Nt = (x[ix].mean() for x in t)
    Mf, Sf, Nf = (x[ix].mean() for x in f)
    rM, rcov, rcoh = Mf / Mt, (Sf / Mf) / (St / Mt), (Nf / Sf) / (Nt / St)
    rN = Nf / Nt
    return rN, rM, rcov, rcoh, Nt / St, Nf / Sf


def main(racine):
    rng = np.random.default_rng(SEED)
    print("Facteurs du rapport N(fort)/N(témoin), fenêtre k = 8–31 ; parts = log facteur / log ρ_N (IC95)")
    for m in sorted(os.listdir(racine)):
        d = os.path.join(racine, m)
        Mt, St, Nt, et = termes(d, "temoin")
        Mf, Sf, Nf, ef = termes(d, "fort")
        P = len(Nt)
        boots = [rng.integers(0, P, P) for _ in range(B)]
        tout = np.arange(P)
        rN, rM, rcov, rcoh, Ct, Cf = facteurs((Mt, St, Nt), (Mf, Sf, Nf), tout)
        bs = np.array([facteurs((Mt, St, Nt), (Mf, Sf, Nf), ix)[:4] for ix in boots])
        parts = np.log(bs[:, 1:4]) / np.log(bs[:, [0]])
        p_pt = np.log([rM, rcov, rcoh]) / np.log(rN)
        c_cov = ci(parts[:, 1])
        if c_cov[0] > 0.5:
            v = "TIENT"
        elif c_cov[0] <= 0 <= c_cov[1] or c_cov[1] < 0.2:
            v = "TOMBE"
        else:
            v = "PARTIELLE"
        print(f"\n== {m}  (vérif. Ah contre A : écart relatif max {max(et, ef):.1e} ; cohérence C témoin {Ct:.3f}, "
              f"fort {Cf:.3f} ; C ≤ 1 : {Ct <= 1 + 1e-6 and Cf <= 1 + 1e-6})")
        print(f"   ρ_N = {rN:.3f} = moyennes ×{rM:.3f} · covariance ×{rcov:.3f} · cohérence ×{rcoh:.3f}")
        for nom, p, j in (("moyennes (attention × valeur)", p_pt[0], 0), ("covariance entre têtes", p_pt[1], 1),
                          ("cohérence (alignement)", p_pt[2], 2)):
            c = ci(parts[:, j])
            print(f"   part {nom:<30} : {p:+.0%}  IC95 [{c[0]:+.0%} ; {c[1]:+.0%}]")
        print(f"   piste « redistribution entre têtes » : {v}")


if __name__ == "__main__":
    main(sys.argv[1])

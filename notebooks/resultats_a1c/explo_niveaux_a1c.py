import sys, numpy as np
C, B = sys.argv[1], sys.argv[2]          # dossiers A1c et A1b
L = lambda d, n: np.load(f"{d}/{n}.npy").astype(float)
print("Niveau d'entropie S_1,45 (moyenne par séquence) — A1b, témoins, jetons 129–223 après t0 :")
for T in ("0.6", "0.8", "1.1"):
    s = L(B, f"S_{T}_temoin")[:, 64 + 129:].mean(1)
    print(f"   T={T} : moyenne {s.mean():.3f} ; quantiles 10/50/90 {np.percentile(s, [10, 50, 90]).round(3)}")
print("A1c volet E (T = 0,8), jetons k = 64–95 après la corruption :")
sid = L(C, "corr_S_identite")[:, 63:95].mean(1); thr = np.percentile(sid, 95)
print(f"   identité : moyenne {sid.mean():.3f} ; seuil haut (95e centile) {thr:.3f}")
for a in ("points", "inversion", "chiffres", "melange", "fautes"):
    s = L(C, f"corr_S_{a}")[:, 63:95].mean(1)
    print(f"   {a:<9} : moyenne {s.mean():.3f} (écart {s.mean() - sid.mean():+.3f}) ; part au-dessus du seuil {np.mean(s > thr):.2f} ; "
          f"écart-type entre prompts {s.std():.3f} (identité {sid.std():.3f})")
s11 = L(B, "S_1.1_temoin")[:, 64 + 129:].mean(1)
print(f"   référence : part des séquences T=1,1 (A1b) au-dessus du même seuil : {np.mean(s11 > thr):.2f}")

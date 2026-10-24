import sys, numpy as np
D = sys.argv[1]
L = lambda n: np.load(f"{D}/{n}.npy").astype(np.float32)
rng = np.random.default_rng(9)
P = 768; B = 300
boots = rng.integers(0, P, (B, P))
for T in ("0.6", "0.8"):
    At, Af, Ad = L(f"Ah_{T}_temoin"), L(f"Ah_{T}_fort"), L(f"Ah_{T}_doux")      # (P, 28, 16)
    def head_ci(X, Y):
        r = X.mean(0) / Y.mean(0)
        bs = np.stack([X[b].mean(0) / Y[b].mean(0) for b in boots])               # (B, 28, 16)
        return r, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    for arm, X in (("fort", Af), ("doux", Ad)):
        r, lo, hi = head_ci(X, At)
        print(f"T={T} {arm}: têtes avec IC95(ρ) < 0,8 : {np.mean(hi < 0.8):.1%} ; IC95 > 1,25 : {np.mean(lo > 1.25):.1%} ; "
              f"couches 0–11 : {np.mean(hi[:12] < 0.8):.1%} < 0,8 / {np.mean(lo[:12] > 1.25):.1%} > 1,25 ; "
              f"couches 16–27 : {np.mean(hi[16:] < 0.8):.1%} / {np.mean(lo[16:] > 1.25):.1%}")
    for meas in ("Alw", "Nlw"):
        Xt, Xf = L(f"{meas}_{T}_temoin")[:, :, 1], L(f"{meas}_{T}_fort")[:, :, 1]
        for name, sl in (("couches 0–11", slice(0, 12)), ("couches 16–27", slice(16, 28))):
            xt, xf = Xt[:, sl].mean(1), Xf[:, sl].mean(1)
            bs = [xf[b].mean() / xt[b].mean() for b in boots]
            print(f"   {meas[0]} {name} : ρ = {xf.mean() / xt.mean():.2f} IC95 [{np.percentile(bs, 2.5):.2f} ; {np.percentile(bs, 97.5):.2f}]")
    # Décroissance : effet direct sur l'entropie contre part d'attention du jeton t0 (témoin et choc)
    R = (L(f"Srej_{T}_fort") - L(f"Srej_{T}_temoin"))[:, 65:]
    At_k, Af_k = L(f"A_{T}_temoin"), L(f"A_{T}_fort")
    for (a, b) in ((2, 3), (16, 31), (64, 95)):
        print(f"   k={a}–{b} : effet direct {R[:, a-1:b].mean():+.4f} ; attention témoin {At_k[:, a-1:b].mean():.5f}, choc {Af_k[:, a-1:b].mean():.5f}")

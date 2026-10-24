import sys, numpy as np
D = sys.argv[1]; T0, GEN = 64, 160
L = lambda n: np.load(f"{D}/{n}.npy").astype(float)
rng = np.random.default_rng(5)
def ci(f, x, B=1000):
    v = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]; return np.percentile(v, [2.5, 97.5])
def tmean(x, p=0.05):
    s = np.sort(x); k = int(p * len(s)); return s[k:len(s) - k].mean()
idx = np.arange(768); j = idx - 64; i = j // 2
is_hand = idx < 64
lang_fr = np.where(is_hand, idx < 32, (j % 2) == 0)
tech = np.where(is_hand, False, (i // 8) >= 24)
for T in ("0.6", "0.8"):
    Rl = (L(f"S_{T}_fort") - L(f"S_{T}_temoin"))[:, T0 + 1:]
    Rr = (L(f"Srej_{T}_fort") - L(f"Srej_{T}_temoin"))[:, T0 + 1:]
    I = (Rl - Rr)[:, 1:31].mean(1); Dd = Rr[:, 1:31].mean(1)
    print(f"=== T={T}  I(k=2–31) moyenne {I.mean():+.4f} {ci(np.mean, I).round(4)} ; médiane {np.median(I):+.4f} {ci(np.median, I).round(4)} ; "
          f"tronquée 5 % {tmean(I):+.4f} {ci(tmean, I).round(4)} ; part I>0 {np.mean(I > 0):.2f}")
    s = np.sort(I)[::-1]
    print(f"   part de la somme portée par les 5 % de prompts au plus grand I : {s[:38].sum() / I.sum():.2f} ; moyenne sans eux {s[38:].mean():+.4f}")
    I2 = (Rl - Rr)[:, 7:31].mean(1)
    print(f"   I(k=8–31) {I2.mean():+.4f} {ci(np.mean, I2).round(4)} (hors complétion immédiate du mot)")
    dsur = L(f"sur_{T}_fort")[:, -64:].mean(1) - L(f"sur_{T}_temoin")[:, -64:].mean(1)
    keep = np.abs(dsur) <= 2.0
    print(f"   prompts à changement de régime (|Δ surprisal fin| > 2 nats) : {np.sum(~keep)} ; I sans eux {I[keep].mean():+.4f} {ci(np.mean, I[keep]).round(4)} ; I chez eux {I[~keep].mean() if (~keep).any() else float('nan'):+.4f}")
    for name, m in (("écrits", is_hand), ("gabarits FR", ~is_hand & lang_fr), ("gabarits EN", ~is_hand & ~lang_fr),
                    ("gabarits généraux", ~is_hand & ~tech), ("gabarits techniques", ~is_hand & tech)):
        print(f"   {name:<20} n={m.sum():3d} I={I[m].mean():+.4f} {ci(np.mean, I[m]).round(4)} D={Dd[m].mean():+.4f}")

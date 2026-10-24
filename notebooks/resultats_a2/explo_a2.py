import sys, numpy as np
D = sys.argv[1]
L = lambda n: np.load(f"{D}/{n}.npy").astype(float)
rng = np.random.default_rng(13); P = 256
boots = [rng.integers(0, P, P) for _ in range(1000)]
H1, H2 = (64, 255), (256, 447)
lev = lambda S, h: S[:, h[0]:h[1] + 1].mean(1)
S0, S75 = L("S_dose_0.0"), L("S_dose_0.75")
d = lev(S75, H2) - lev(S0, H2)
c = np.percentile([d[b].mean() for b in boots], [2.5, 97.5])
print(f"Addendum point fixe : Δ = S̄(H2, 0,75) − S̄(H2, 0) = {d.mean():+.3f} IC95 [{c[0]:+.3f} ; {c[1]:+.3f}] → "
      + ("POINT FIXE" if c[0] >= -0.05 and c[1] <= 0.05 else "PAS DE POINT FIXE" if c[0] > 0.05 else "INDÉTERMINÉ"))
def roll(S, a, b, W=32):
    x = S[:, a:b + 1]; cs = np.cumsum(np.pad(x, ((0, 0), (1, 0))), 1); return (cs[:, W:] - cs[:, :-W]) / W
print("Bascule absorbante ? (séquences au-dessus de 1,6 pendant H1 → encore au-dessus sur les 32 derniers pas)")
for name in ("S_dose_0.35", "S_dose_0.5", "S_dose_0.75", "S_T_1.0", "S_T_1.1"):
    S = L(name); up1 = roll(S, 64, 255).max(1) > 1.6; end = S[:, 416:448].mean(1) > 1.6
    print(f"   {name:<12} : basculées en H1 {up1.mean():.2f} ; parmi elles, encore hautes à la fin {end[up1].mean() if up1.any() else float('nan'):.2f} ; "
          f"non basculées en H1 mais hautes à la fin {end[~up1].mean():.2f}")
print("Répartition des niveaux H2 par séquence (même entropie moyenne : dose 0,5 contre T = 1,0 ; dose 0,2 contre T = 0,9)")
for name in ("S_dose_0.5", "S_T_1.0", "S_dose_0.2", "S_T_0.9", "S_dose_0.0", "S_T_0.8"):
    x = lev(L(name), H2)
    print(f"   {name:<12} : moyenne {x.mean():.3f} ; écart-type {x.std():.3f} ; part < 0,6 : {np.mean(x < 0.6):.2f} ; part > 1,6 : {np.mean(x > 1.6):.2f} ; "
          f"quantiles 10/50/90 {np.percentile(x, [10, 50, 90]).round(2)}")
print("Niveau absolu H1 → H2 (moyenne sur prompts) : la dose compense-t-elle la baisse naturelle ?")
for p in ("0.0", "0.03", "0.1", "0.2", "0.35", "0.5", "0.75"):
    S = L(f"S_dose_{p}"); x = lev(S, H2) - lev(S, H1); ci = np.percentile([x[b].mean() for b in boots], [2.5, 97.5])
    print(f"   dose {p:<4} : H2 − H1 = {x.mean():+.3f} IC95 [{ci[0]:+.3f} ; {ci[1]:+.3f}]")

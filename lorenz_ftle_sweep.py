"""Balayage de la fenêtre FTLE sur plusieurs conditions initiales.

Pour chaque condition initiale (CI) et chaque fenêtre T :
  Spearman(lam_S, FTLE_T) et Spearman(g_inst, FTLE_T).
On trace la moyenne ± écart-type entre CI, et on teste si l'oscillation
de la corrélation en fonction de T est systématique (mêmes pics/creux pour
toutes les CI) ou du bruit propre à une trajectoire.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lorenz_stretching import dt, integrate, ftle_centered, spearman

N_IC = 10
SEED = 0
T_grid = np.round(np.arange(0.1, 5.0001, 0.1), 2)
OUT_PNG = "lorenz_ftle_sweep.png"
OUT_CSV = "lorenz_ftle_sweep.csv"
# Paires (creux, pic) repérées sur la CI unique [1,1,1] ; testées ici de façon appariée
CONTRASTS = [(1.2, 2.0), (3.0, 2.0), (3.0, 5.0)]

rng = np.random.default_rng(SEED)
ics = rng.uniform([-20, -25, 5], [20, 25, 45], size=(N_IC, 3))
deltas0 = rng.normal(size=(N_IC, 3))

rho_S = np.zeros((N_IC, len(T_grid)))
rho_g = np.zeros((N_IC, len(T_grid)))
lambda1 = np.zeros(N_IC)
periods = np.zeros(N_IC)

for k in range(N_IC):
    traj, lam_S, g_step, g_inst = integrate(ics[k], deltas0[k])
    lambda1[k] = g_step.mean()
    # Période dominante de lam_S (pic du spectre hors fréquence nulle)
    spec = np.abs(np.fft.rfft(lam_S - lam_S.mean()))
    freqs = np.fft.rfftfreq(len(lam_S), dt)
    periods[k] = 1 / freqs[1:][np.argmax(spec[1:])]
    for j, T in enumerate(T_grid):
        ftle, valid = ftle_centered(g_step, T)
        rho_S[k, j] = spearman(lam_S[valid], ftle[valid])
        rho_g[k, j] = spearman(g_inst[valid], ftle[valid])
    print(f"CI {k}: x0={np.round(ics[k], 1)}  lambda_1={lambda1[k]:.3f}  "
          f"période dominante de lam_S={periods[k]:.3f}")

mS, sS = rho_S.mean(0), rho_S.std(0, ddof=1)
mg, sg = rho_g.mean(0), rho_g.std(0, ddof=1)

np.savetxt(OUT_CSV, np.column_stack([T_grid, mS, sS, mg, sg, rho_S.T, rho_g.T]),
           delimiter=",", fmt="%.4f",
           header="T,mean_rhoS,std_rhoS,mean_rhog,std_rhog,"
                  + ",".join(f"rhoS_ci{k}" for k in range(N_IC)) + ","
                  + ",".join(f"rhog_ci{k}" for k in range(N_IC)), comments="")

print(f"\nlambda_1 : {lambda1.mean():.3f} ± {lambda1.std(ddof=1):.3f}")
print(f"\n{'T':>5} {'rho(lamS,FTLE)':>18} {'rho(g,FTLE)':>18}")
for T in [0.1, 0.25, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
    j = int(np.argmin(np.abs(T_grid - T)))
    print(f"{T:>5} {mS[j]:+.3f} ± {sS[j]:.3f}     {mg[j]:+.3f} ± {sg[j]:.3f}")

T_cross = T_grid[np.argmax(mS < 0.5)]
print(f"\nPremier T où la moyenne de rho(lam_S, FTLE) passe sous 0.5 : {T_cross}")

print("\nContrastes appariés rho_S(T_b) - rho_S(T_a) (moyenne ± SE, nb de CI où > 0) :")
for Ta, Tb in CONTRASTS:
    ja, jb = (int(np.argmin(np.abs(T_grid - t))) for t in (Ta, Tb))
    d = rho_S[:, jb] - rho_S[:, ja]
    se = d.std(ddof=1) / np.sqrt(N_IC)
    print(f"  T={Tb} vs T={Ta} : {d.mean():+.3f} ± {se:.3f}  (t={d.mean() / se:+.1f}, "
          f"{np.sum(d > 0)}/{N_IC} CI positives)")

# Figure : moyenne ± écart-type, courbes individuelles en fond
C1, C2 = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(10, 5.5))
for k in range(N_IC):
    ax.plot(T_grid, rho_S[k], color=C1, lw=0.6, alpha=0.25)
    ax.plot(T_grid, rho_g[k], color=C2, lw=0.6, alpha=0.25)
ax.fill_between(T_grid, mS - sS, mS + sS, color=C1, alpha=0.15, lw=0)
ax.fill_between(T_grid, mg - sg, mg + sg, color=C2, alpha=0.15, lw=0)
ax.plot(T_grid, mS, color=C1, lw=2, label="λmax(S) vs FTLE")
ax.plot(T_grid, mg, color=C2, lw=2, label="uᵀSu vs FTLE")
ax.axhline(0, color="#888888", lw=0.8)
ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
ax.text(T_grid[-1], 0.52, "seuil 0,5", ha="right", va="bottom", fontsize=8, color="#555555")
# Étiquettes directes en bout de courbe (encre texte, pas la couleur de série)
ax.text(T_grid[-1] + 0.06, mS[-1], "λmax(S)", color="#333333", fontsize=9, va="center")
ax.text(T_grid[-1] + 0.06, mg[-1], "uᵀSu", color="#333333", fontsize=9, va="center")
ax.set_xlim(0, T_grid[-1] + 0.55)
ax.set_xlabel("Fenêtre FTLE T (unités de temps)")
ax.set_ylabel("Corrélation de Spearman avec le FTLE")
ax.set_title(f"Lorenz — {N_IC} conditions initiales : moyenne ± écart-type "
             f"(λ₁ = {lambda1.mean():.3f})")
ax.set_ylim(-0.6, 1.05)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print(f"\nFigure : {OUT_PNG}   Données : {OUT_CSV}")

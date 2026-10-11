"""Étirement local sur l'attracteur de Lorenz : borne instantanée vs taux réalisé.

Trois mesures, de la plus locale à la plus robuste :
  1. lam_S  : valeur propre max de S = (J + J^T)/2. Borne supérieure du taux
              d'étirement instantané de N'IMPORTE QUELLE direction. Dépend des
              coordonnées (non invariante) et ignore la direction réellement suivie.
  2. g_inst : taux d'étirement instantané du vecteur tangent effectivement porté
              par le flot (vecteur de Lyapunov « arrière ») : g = u^T S u, u = delta/|delta|.
              Toujours g_inst <= lam_S.
  3. ftle   : moyenne de g_inst sur une fenêtre T (exposant de Lyapunov à temps fini).
              Sa moyenne sur toute la trajectoire doit redonner lambda_1 ~ 0.906.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paramètres de Lorenz (régime chaotique)
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
dt = 0.005
N_steps = 40000
cut = 5000          # transitoire (trajectoire + alignement du vecteur tangent)
T_ftle = 1.0        # fenêtre FTLE en unités de temps
OUT = "lorenz_stretching.png"


def lorenz(s):
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def jacobian(s):
    x, y, z = s
    return np.array([[-sigma, sigma, 0.0],
                     [rho - z, -1.0, -x],
                     [y, x, -beta]])


def rk4_step(s, d):
    """RK4 couplé état + équation linéarisée d(delta)/dt = J(s(t)) delta."""
    k1s = lorenz(s);                 k1d = jacobian(s) @ d
    s2 = s + 0.5 * dt * k1s;         k2s = lorenz(s2); k2d = jacobian(s2) @ (d + 0.5 * dt * k1d)
    s3 = s + 0.5 * dt * k2s;         k3s = lorenz(s3); k3d = jacobian(s3) @ (d + 0.5 * dt * k2d)
    s4 = s + dt * k3s;               k4s = lorenz(s4); k4d = jacobian(s4) @ (d + dt * k3d)
    return (s + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s),
            d + dt / 6 * (k1d + 2 * k2d + 2 * k3d + k4d))


traj = np.zeros((N_steps, 3))
lam_S = np.zeros(N_steps)
g_step = np.zeros(N_steps)   # ln(|delta_{i+1}| / |delta_i|) / dt, delta_i unitaire
g_inst = np.zeros(N_steps)   # u^T S u au point i

state = np.array([1.0, 1.0, 1.0])
delta = np.array([1.0, 0.0, 0.0])

for i in range(N_steps):
    traj[i] = state
    S = 0.5 * (jacobian(state) + jacobian(state).T)
    lam_S[i] = np.linalg.eigvalsh(S)[-1]
    g_inst[i] = delta @ S @ delta
    state, delta = rk4_step(state, delta)
    n = np.linalg.norm(delta)
    g_step[i] = np.log(n) / dt
    delta /= n                   # renormalisation de Benettin à chaque pas

traj, lam_S, g_step, g_inst = (a[cut:] for a in (traj, lam_S, g_step, g_inst))

# FTLE centré sur une fenêtre glissante de T_ftle
w = int(round(T_ftle / dt))
ftle = np.convolve(g_step, np.ones(w) / w, mode="same")
valid = slice(w // 2, len(ftle) - w // 2)

lambda1 = g_step.mean()
frac_neg_ftle = np.mean(ftle[valid] < 0)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


rho_S_ftle = spearman(lam_S[valid], ftle[valid])
rho_g_ftle = spearman(g_inst[valid], ftle[valid])

print(f"lambda_1 estimé (moyenne de g)        : {lambda1:.3f}  (référence ~0.906)")
print(f"lam_S  : min {lam_S.min():.2f}  médiane {np.median(lam_S):.2f}  max {lam_S.max():.2f}")
print(f"g_inst : min {g_inst.min():.2f}  médiane {np.median(g_inst):.2f}  max {g_inst.max():.2f}")
print(f"fraction g_inst > lam_S (doit être 0) : {np.mean(g_inst > lam_S + 1e-9):.4f}")
print(f"fraction de points avec FTLE(T={T_ftle}) < 0 : {frac_neg_ftle:.3f}")
print(f"Spearman(lam_S, FTLE)  = {rho_S_ftle:+.3f}")
print(f"Spearman(g_inst, FTLE) = {rho_g_ftle:+.3f}")

# Points fixes : C± sont des selles-foyers (instables pour rho > ~24.74)
C_plus = np.array([np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1])
fixed = {"Origine (selle)": (np.zeros(3), "cyan", "x"),
         "C+ (selle-foyer)": (C_plus, "lime", "*"),
         "C− (selle-foyer)": (C_plus * [-1, -1, 1], "lime", "*")}

fig = plt.figure(figsize=(16, 6))
panels = [(lam_S, "λmax(S) : borne instantanée\n(non invariante, ignore la direction)", "inferno", None),
          (ftle, f"FTLE (T = {T_ftle}) : étirement réellement réalisé\n(moyenne = {lambda1:.3f})", "coolwarm", 0.0)]
tr = traj[valid]
for k, (val, title, cmap, center) in enumerate(panels, start=1):
    ax = fig.add_subplot(1, 2, k, projection="3d")
    v = val[valid]
    if center is None:
        norm = matplotlib.colors.Normalize(*np.percentile(v, [1, 99]))
    else:
        m = np.percentile(np.abs(v), 99)
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=-m, vmax=m)
    sc = ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2], c=v, cmap=cmap, norm=norm, s=0.3, alpha=0.8)
    for lab, (p, col, mk) in fixed.items():
        ax.scatter(*p, c=col, s=80, marker=mk, label=lab)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="taux (1/unité de temps)")
    if k == 1:
        ax.legend(loc="upper left", fontsize=8)

fig.suptitle(f"Lorenz — Spearman(λmax(S), FTLE) = {rho_S_ftle:+.2f}")
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print(f"Figure écrite : {OUT}")

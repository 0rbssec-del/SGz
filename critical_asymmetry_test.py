"""Banc d'essai : contrôleur original vs corrigé vs p fixe, sur le modèle jouet.

1. Grille statique A(T, p) et composantes : vérifie le sens de variation de A
   avec p (signe du gain de l'actionneur), pour les deux modes de mesure.
2. Boucle fermée sur N graines : erreur |A_lisse - A*|, fraction dans [0.4, 0.6],
   fraction du temps où p est en butée.
"""
import numpy as np

from critical_asymmetry import (CorrectedController, OriginalController, ToyBigramModel,
                                generate, measure_A)

N_SEEDS, N_TOK, A_STAR = 10, 300, 0.5
OUT_CSV = "critical_asymmetry_results.csv"


class FixedController:
    def __init__(self):
        self.T, self.p = 1.0, 0.85

    def step(self, A_lisse):
        pass

    def after_sample(self, A_lisse):
        pass


model = ToyBigramModel(seed=0)
rng = np.random.default_rng(1)
ctxs = [[int(c)] for c in rng.integers(model.V, size=50)]
MODES = ["echantillon", "modele"]
for mesure in MODES:
    print(f"Grille statique, mesure = {mesure} (moyenne sur 50 contextes) : A [phi1 phi2 phi6]")
    for T in [0.7, 1.0, 1.3]:
        cells = []
        for p in [0.5, 0.8, 0.99]:
            r = np.mean([measure_A(model, c, T, p, mesure)[:4] for c in ctxs], axis=0)
            cells.append(f"p={p:.2f}: {r[0]:.3f} [{r[1]:.2f} {r[2]:.2f} {r[3]:.2f}]")
        print(f"  T={T:.1f}  " + "   ".join(cells))

lines = ["mesure,controleur,graine,err_moy,frac_zone,frac_butee_p,A_final,p_final,T_final"]
print(f"\nBoucle fermée, A* = {A_STAR}, {N_SEEDS} graines x {N_TOK} tokens (après 30 tokens de rodage)")
for mesure, name, cls in [(m, n, c) for m in MODES for n, c in
                          [("fixe", FixedController), ("original", OriginalController),
                           ("corrige", CorrectedController)]]:
    stats = []
    for s in range(N_SEEDS):
        r = generate(model, cls(), n_tokens=N_TOK, seed=s, mesure=mesure)[30:]
        A, p, T = r[:, 2], r[:, 7], r[:, 6]
        st = (np.mean(np.abs(A - A_STAR)), np.mean((A >= 0.4) & (A <= 0.6)),
              np.mean((p <= 0.5) | (p >= 0.99)), A[-1], p[-1], T[-1])
        stats.append(st)
        lines.append(f"{mesure},{name},{s}," + ",".join(f"{v:.4f}" for v in st))
    m, sd = np.mean(stats, axis=0), np.std(stats, axis=0)
    print(f"  {mesure:10s} {name:9s} err = {m[0]:.3f}±{sd[0]:.3f}  zone = {m[1]:.2f}  butée p = {m[2]:.2f}"
          f"  A_fin = {m[3]:.3f}  p_fin = {m[4]:.2f}  T_fin = {m[5]:.2f}")

with open(OUT_CSV, "w") as f:
    f.write("\n".join(lines) + "\n")

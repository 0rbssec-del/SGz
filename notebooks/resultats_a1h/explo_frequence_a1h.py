import sys, numpy as np
from scipy.stats import spearmanr
D = sys.argv[1]; L = lambda n: np.load(f"{D}/{n}.npy")
tok = L("tok"); T0 = 64
counts = np.bincount(tok.ravel(), minlength=int(tok.max()) + 200000)
w = lambda X: X[:, 7:31].mean(1)
nt = w(L("N_temoin")).astype(float)
rng = np.random.default_rng(1)
for arm in ("fort", "FI", "RA"):
    t = L(f"tok0_{arm}"); c = counts[t]
    r = np.log(w(L(f"N_{arm}")).astype(float) / nt)
    rho, p = spearmanr(np.log1p(c), r)
    lo, hi = np.median(c), None
    lowf, highf = r[c <= np.median(c)], r[c > np.median(c)]
    print(f"   {arm:<4} : fréquence médiane {np.median(c):.0f} (max {c.max()}) ; Spearman(log fréquence, log ρ par prompt) = {rho:+.3f} (p = {p:.3f}) ; "
          f"ρ géométrique moitié rare {np.exp(lowf.mean()):.2f}, moitié fréquente {np.exp(highf.mean()) if highf.size else float('nan'):.2f}")

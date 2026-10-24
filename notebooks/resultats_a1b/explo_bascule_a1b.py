import sys, numpy as np
D = sys.argv[1]; T0, GEN, W = 64, 288, 32
L = lambda n: np.load(f"{D}/{n}.npy").astype(float)
def roll(x):  # moyenne glissante de longueur W, alignée à gauche
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0))), 1); return (c[:, W:] - c[:, :-W]) / W
rng = np.random.default_rng(11)
for T in ("0.8", "1.1"):
    st, sf = L(f"sur_{T}_temoin"), L(f"sur_{T}_fort")
    lastt, lastf = st[:, -64:].mean(1), sf[:, -64:].mean(1)
    print(f"T={T} surprisal moyenne des 64 derniers jetons : témoin quantiles 10/50/90 {np.percentile(lastt,[10,50,90]).round(2)} ; choc {np.percentile(lastf,[10,50,90]).round(2)}")
    base = st[:, :T0].mean(1)   # même préfixe pour les deux bras
    def date(s):
        r = roll(s); hit = r[:, T0+1:] > base[:, None] + 2.0
        d = np.where(hit.any(1), hit.argmax(1) + T0 + 1, -1); return d
    dt, df = date(st), date(sf)
    both = (dt >= 0) & (df >= 0)
    print(f"   bascule (moyenne sur {W} jetons > préfixe + 2 nats) : témoin {np.mean(dt>=0):.3f}, choc {np.mean(df>=0):.3f}, les deux {both.mean():.3f}")
    if both.sum() >= 10:
        dd = (df - dt)[both]; v = [np.median(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(1000)]
        print(f"   paires où les deux basculent (n={both.sum()}) : date choc − témoin médiane {np.median(dd):+.0f} jetons IC95 {np.percentile(v,[2.5,97.5])} ; part où le choc bascule plus tôt {np.mean(dd<0):.2f}, plus tard {np.mean(dd>0):.2f}")
        print(f"   dates témoin (quantiles 10/50/90) : {np.percentile(dt[both],[10,50,90])}")

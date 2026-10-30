"""A1z2 après coup : d'où vient l'écart ρ_a(choc) − ρ_a(témoin) ?
(1) composantes absolues et leurs rapports ; (2) part des prompts où les premiers jetons diffèrent contre effet par prompt
concerné ; (3) pente intra-prompt de l'entropie qui suit sur la surprise du premier jeton ; (4) IC par grappes de thème."""
# Usage : python3 apres_coup.py <dossier contenant q/ (Qwen3) et s/ (SmolLM3), contenu des archives *_a1z2.zip>
import sys
import numpy as np

D = sys.argv[1]
WF = slice(3, 32)                                   # fenêtre du plan factoriel, k = 3–31
B = 1000
n = 768
# Structure des 768 prompts (code A1c–A1z, voir resultats_a1z/strates.py)
main = np.arange(n) < 64
lang = np.array(["fr" if (i < 32 or (i >= 64 and (i - 64) % 2 == 0)) else "en" for i in range(n)])
m_ = np.where(main, -1, (np.arange(n) - 64) // 2)
grappe = np.where(main, np.arange(n), 1000 + (m_ // 8) * 2 + (lang == "en"))


def carres(Y):
    I, J = Y.shape[1:]
    m = Y.mean((1, 2), keepdims=True)
    ri, cj = Y.mean(2, keepdims=True) - m, Y.mean(1, keepdims=True) - m
    res = Y - m - ri - cj
    return np.stack([J * (ri ** 2).sum((1, 2)) / (I - 1), I * (cj ** 2).sum((1, 2)) / (J - 1),
                     (res ** 2).sum((1, 2)) / ((I - 1) * (J - 1))], 1)


def parts(K):
    msa, msb, mse = K.mean(0)
    sa, sb = (msa - mse) / 3, (msb - mse) / 3
    tot = sa + sb + mse
    return np.array([sa / tot, sb / tot, sa, sb, mse])


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5])


def boots_naifs(rng):
    return [rng.integers(0, n, n) for _ in range(B)]


def boots_grappes(rng):
    u = np.unique(grappe)
    idx = {k: np.nonzero(grappe == k)[0] for k in u}
    return [np.concatenate([idx[k] for k in rng.choice(u, len(u))]) for _ in range(B)]


def pente(Y, S):
    """Pente intra-prompt : moyenne sur le bruit de l'entropie qui suit (Y[p, i, :]) contre surprise du premier jeton i (S[i, p])."""
    y = Y.mean(2) - Y.mean((1, 2))[:, None]
    s = S.T - S.T.mean(1, keepdims=True)
    return lambda ix: float((y[ix] * s[ix]).sum() / (s[ix] ** 2).sum())


fm = lambda c: f"[{c[0]:+.4f} ; {c[1]:+.4f}]"
for lab, sub in (("Qwen3-1.7B-Base", "q"), ("SmolLM3-3B-Base", "s")):
    L = lambda k: np.load(f"{D}/{sub}/{k}.npy", allow_pickle=False)
    moy = lambda k: (L("S_temoin") if k == "w0n0" else L(f"S_{k}"))[:, WF].mean(1).astype(float)
    Y = {c: np.stack([np.stack([moy(f"{c}{i}n{j}") for j in range(3)], 1) for i in range(3)], 1) for c in "cw"}
    K = {c: carres(Y[c]) for c in "cw"}
    PC, PW, SC, SW = L("PC"), L("PW"), L("SC").astype(float), L("SW").astype(float)
    ndist = {"c": np.array([len(set(PC[:, p])) for p in range(n)]), "w": np.array([len(set(PW[:, p])) for p in range(n)])}
    rng = np.random.default_rng(7)
    bn, bg = boots_naifs(rng), boots_grappes(rng)
    pc, pt = parts(K["c"]), parts(K["w"])
    print(f"\n=== {lab}")
    print(f"[CONTRÔLE] ρ_a choc {pc[0]:.3f}, témoin {pt[0]:.3f} (doit égaler resume_a1z2.json)")

    # (1) Composantes absolues : rapports choc / témoin
    for j, nom in ((2, "σ²_a premier jeton"), (3, "σ²_b bruit"), (4, "σ²_e reste")):
        r = pc[j] / pt[j]
        c = ci([parts(K["c"][ix])[j] / parts(K["w"][ix])[j] for ix in bn])
        print(f"[1] {nom:<20}: choc {pc[j]:.2e}, témoin {pt[j]:.2e}, rapport ×{r:.2f} IC95 [{c[0]:.2f} ; {c[1]:.2f}]")
    tot = lambda p: p[2] + p[3] + p[4]
    c = ci([tot(parts(K["c"][ix])) / tot(parts(K["w"][ix])) for ix in bn])
    print(f"[1] variance totale entre les 9 suites : rapport ×{tot(pc) / tot(pt):.2f} IC95 [{c[0]:.2f} ; {c[1]:.2f}]")

    # (2) σ²_a = (part des prompts où les premiers jetons diffèrent) × (σ²_a parmi ces prompts) : exact, car un prompt aux
    # trois premiers jetons identiques donne MS_a = MS_e = 0
    out = {}
    for c_ in "cw":
        d = ndist[c_] > 1
        out[c_] = (d.mean(), parts(K[c_][d])[2])
        print(f"[2] {'choc' if c_ == 'c' else 'témoin':<6}: premiers jetons différents dans {d.mean():.0%} des prompts "
              f"(2 distincts {np.mean(ndist[c_] == 2):.0%}, 3 distincts {np.mean(ndist[c_] == 3):.0%}) ; σ²_a parmi eux "
              f"{out[c_][1]:.2e}")
    rf, rs = out["c"][0] / out["w"][0], out["c"][1] / out["w"][1]
    print(f"[2] rapport σ²_a ×{rf * rs:.2f} = ×{rf:.2f} (plus de prompts concernés) × ×{rs:.2f} (effet par prompt concerné) ; "
          f"part du log-rapport due à la diversité des premiers jetons : {np.log(rf) / np.log(rf * rs):.0%}")
    # Même nombre de premiers jetons distincts : ρ_a choc − témoin par strate, puis témoin repondéré sur la répartition du choc
    for k in (2, 3):
        mc, mw = ndist["c"] == k, ndist["w"] == k
        if mc.sum() > 20 and mw.sum() > 20:
            print(f"[2] strate {k} premiers jetons distincts : ρ_a choc {parts(K['c'][mc])[0]:.3f} (n = {mc.sum()}), "
                  f"témoin {parts(K['w'][mw])[0]:.3f} (n = {mw.sum()})")

    def dr_strat(ix):
        Kc, Kw, nc, nw = K["c"][ix], K["w"][ix], ndist["c"][ix], ndist["w"][ix]
        Kw_rep = np.zeros(3)
        for k in (1, 2, 3):
            if (nw == k).any():
                Kw_rep += (nc == k).mean() * Kw[nw == k].mean(0)
        return parts(Kc)[0] - parts(Kw_rep[None])[0]
    c = ci([dr_strat(ix) for ix in bn])
    print(f"[2] Δρ avec le témoin repondéré sur la répartition du choc (1, 2 ou 3 premiers jetons distincts) : "
          f"{dr_strat(np.arange(n)):+.3f} IC95 [{c[0]:+.3f} ; {c[1]:+.3f}] (Δρ brut {pc[0] - pt[0]:+.3f})")

    # (3) Le premier jeton le plus surprenant est-il suivi de plus d'incertitude ? (pente intra-prompt, nats par nat)
    for c_, S in (("c", SC), ("w", SW)):
        f = pente(Y[c_], S)
        print(f"[3] {'choc' if c_ == 'c' else 'témoin':<6}: pente intra-prompt entropie (k = 3–31) / surprise du premier jeton "
              f"{f(np.arange(n)):+.4f} IC95 {fm(ci([f(ix) for ix in bn]))}")

    # (4) IC par grappes de thème pour les deux quantités de décision
    for nom, fn in (("Δρ (Z2a)", lambda ix: parts(K["c"][ix])[0] - parts(K["w"][ix])[0]),
                    ("ρ_a choc (Z2b)", lambda ix: parts(K["c"][ix])[0])):
        a, g = ci([fn(ix) for ix in bn]), ci([fn(ix) for ix in bg])
        print(f"[4] {nom:<15}: IC95 naïf [{a[0]:+.3f} ; {a[1]:+.3f}] ; par grappes de thème [{g[0]:+.3f} ; {g[1]:+.3f}] "
              f"(×{(g[1] - g[0]) / (a[1] - a[0]):.2f})")

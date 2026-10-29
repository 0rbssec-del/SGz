"""A1z après coup : h(1) et T(b(1))/T(apparié) selon la surprise du premier jeton du choc dans la loi du témoin."""
# Usage : python3 apres_coup.py <dossier contenant q/ (Qwen3) et s/ (SmolLM3), contenu des archives *_a1z.zip>
import sys
import numpy as np

D = sys.argv[1]
T1 = 64
W = slice(2, 32)
rng = np.random.default_rng(20261025 + 3)


def ci(v):
    v = np.asarray(v, float)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def ratio(x, y, B=1000):
    bs = [rng.integers(0, len(x), len(x)) for _ in range(B)]
    return x.mean() / y.mean(), ci([x[i].mean() / y[i].mean() for i in bs])


for lab, sub in (("Qwen3-1.7B-Base", "q"), ("SmolLM3-3B-Base", "s")):
    L = lambda k: np.load(f"{D}/{sub}/{k}.npy", allow_pickle=False)
    m = lambda k: L(f"S_{k}")[:, W].mean(1).astype(float)
    St, Sr, Sc, Sb1, Sap, Sa1 = m("temoin"), m("rejoue"), m("choc"), m("b1"), m("bapp"), m("a1")
    T0, Tb1, Tap, I0, Ia1 = Sc - St, Sb1 - St, Sap - St, Sc - Sr, Sa1 - Sr
    sc, sa, rk = L("surp_choc"), L("surp_app"), L("rang_choc")
    tt, tc = L("tok_temoin"), L("tok_choc")
    meme = tc[:, T1 + 1] == tt[:, T1 + 1]
    print(f"\n=== {lab} : {len(T0)} prompts ; premier jeton du choc = celui du témoin : {meme.mean():.1%}")
    print(f"surprise du premier jeton du choc dans la loi du témoin (s_c) : quartiles {np.percentile(sc[~meme], [25, 50, 75]).round(2)} nats ; "
          f"rang médian {np.median(rk[~meme]):.0f} ; s_c < 3 nats : {np.mean(sc[~meme] < 3):.1%}")
    print(f"{'strate':<26}{'n':>5}{'T0':>9}{'h(1)':>8}{'IC95':>18}{'T(b1)/T(app)':>14}{'IC95':>18}")
    for nom, msk in (("tous", np.ones(len(T0), bool)), ("premier jeton identique", meme),
                     ("s_c < 3 nats", ~meme & (sc < 3)), ("3 ≤ s_c < 10", ~meme & (sc >= 3) & (sc < 10)),
                     ("s_c ≥ 10", ~meme & (sc >= 10))):
        if msk.sum() < 20:
            print(f"{nom:<26}{msk.sum():>5}  (trop peu)")
            continue
        h, hc = ratio(Tb1[msk], T0[msk])
        r, rc = ratio(Tb1[msk], Tap[msk]) if not nom.startswith("premier") else (np.nan, (np.nan, np.nan))
        print(f"{nom:<26}{msk.sum():>5}{T0[msk].mean():>+9.4f}{h:>8.2f}   [{hc[0]:+.2f} ; {hc[1]:+.2f}]{r:>12.2f}   [{rc[0]:.2f} ; {rc[1]:.2f}]")
    g, gc_ = ratio(Ia1[~meme], I0[~meme])
    print(f"g(1) sans les prompts où le premier jeton est inchangé : {g:+.2f} IC95 [{gc_[0]:+.2f} ; {gc_[1]:+.2f}]")
    rs = lambda x, y: np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
    print(f"Spearman(s_c, T(b1)) = {rs(sc[~meme], Tb1[~meme]):+.3f} ; Spearman(T0, T(b1)) = {rs(T0, Tb1):+.3f} ; "
          f"Spearman(T(app), T(b1)) = {rs(Tap[~meme], Tb1[~meme]):+.3f}")


print("\n--- Expérience naturelle : prompts où le choc ne change pas le premier jeton choisi ---")
for lab, sub in (("Qwen3-1.7B-Base", "q"), ("SmolLM3-3B-Base", "s")):
    L = lambda k: np.load(f"{D}/{sub}/{k}.npy", allow_pickle=False)
    Sx = {k: L(f"S_{k}").astype(float) for k in ("temoin", "rejoue", "choc")}
    m = lambda k: Sx[k][:, W].mean(1)
    St, Sr, Sc = m("temoin"), m("rejoue"), m("choc")
    D0, I0 = Sr - St, Sc - Sr
    tt, tc = L("tok_temoin"), L("tok_choc")
    meme = tc[:, T1 + 1] == tt[:, T1 + 1]
    d1 = Sx["rejoue"][:, 1] - Sx["temoin"][:, 1]          # effet immédiat : entropie de la loi du jeton t₀ + 1
    for nom, msk in (("premier jeton inchangé", meme), ("premier jeton changé", ~meme)):
        bs = [rng.integers(0, msk.sum(), msk.sum()) for _ in range(1000)]
        x, y = I0[msk], D0[msk]
        print(f"{lab:<16} {nom:<24} n={msk.sum():>3} : D₀ {y.mean():+.4f} ; I₀ {x.mean():+.4f} IC95 [{ci([x[i].mean() for i in bs])[0]:+.4f} ; "
              f"{ci([x[i].mean() for i in bs])[1]:+.4f}] ; I₀/D₀ {x.mean() / y.mean():+.2f} ; effet immédiat k = 1 : {d1[msk].mean():+.3f}")
    rs = lambda a, b: np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    print(f"{lab:<16} Spearman(effet immédiat k = 1, I₀) = {rs(d1, I0):+.3f} ; Spearman(effet immédiat, D₀) = {rs(d1, D0):+.3f}")

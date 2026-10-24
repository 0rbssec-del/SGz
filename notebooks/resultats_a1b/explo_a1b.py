import sys, numpy as np
D = sys.argv[1]
T0, GEN = 64, 288
L = lambda n: np.load(f"{D}/{n}.npy")
def R(T, a): return (L(f"S_{T}_{a}")[:, T0+1:] - L(f"S_{T}_temoin")[:, T0+1:]).astype(float)
def dsur(T, a): return (L(f"sur_{T}_{a}")[:, GEN-64:].mean(1) - L(f"sur_{T}_temoin")[:, GEN-64:].mean(1))
def win(r, w): return r[:, w[0]-1:w[1]].mean(1)
rng = np.random.default_rng(7)
def ci(f, *arrs, B=1000):
    P = arrs[0].shape[0]; v = []
    for _ in range(B):
        ix = rng.integers(0, P, P); v.append(f(*[a[ix] for a in arrs]))
    return np.percentile(v, [2.5, 97.5])
EARLY, MIDRL, LATE = (1, 16), (33, 127), (129, 223)
hand, tmpl = np.arange(64), np.arange(64, 256)
print("1. B1 par jeu de prompts (E, R_tard, marge=0,3E, IC95 R_tard)")
for T in ("0.6", "0.8"):
    r = R(T, "fort")
    for name, ix in (("écrits(64)", hand), ("gabarits(192)", tmpl), ("tous(256)", np.arange(256))):
        E, lt = win(r[ix], EARLY), win(r[ix], LATE)
        c = ci(np.mean, lt)
        print(f"   T={T} {name:<14} E={E.mean():+.4f}  R_tard={lt.mean():+.4f} IC95[{c[0]:+.4f};{c[1]:+.4f}]  marge ±{0.3*E.mean():.4f}  R_tard/E={lt.mean()/E.mean():+.2f}")
print("2. R_tard à T=0,8 (tous) : médiane, moyenne tronquée 5 %, part des prompts > 0")
lt = win(R("0.8", "fort"), LATE); s = np.sort(lt); k = int(0.05*len(s))
print(f"   médiane {np.median(lt):+.4f} ; tronquée {s[k:-k].mean():+.4f} ; part>0 {np.mean(lt>0):.2f} ; IC95 médiane {ci(np.median, lt)}")
print("3. Persistance relative P/E (k=33–127 ÷ k=1–16), et Δ RL hors prompts à bascule (|Δsur|>2 nats à T=1,1, exclusion symétrique)")
r8, r11 = R("0.8", "fort"), R("1.1", "fort")
for T, r in (("0.8", r8), ("1.1", r11)):
    print(f"   T={T} P={win(r,MIDRL).mean():+.4f} E={win(r,EARLY).mean():+.4f} P/E={win(r,MIDRL).mean()/win(r,EARLY).mean():+.2f}")
f = lambda a, b: win(b, MIDRL).mean()/win(b, EARLY).mean() - win(a, MIDRL).mean()/win(a, EARLY).mean()
print(f"   Δ(P/E) 1,1 − 0,8 = {f(r8, r11):+.3f} IC95 {ci(f, r8, r11)}")
d11 = dsur("1.1", "fort"); keep = np.abs(d11) <= 2.0
g = lambda a, b: (win(b, MIDRL) - win(a, MIDRL)).mean()
print(f"   prompts gardés {keep.sum()}/256 ; Δ = {g(r8[keep], r11[keep]):+.4f} IC95 {ci(g, r8[keep], r11[keep])}")
print(f"   (tous) Δ = {g(r8, r11):+.4f} IC95 {ci(g, r8, r11)}")
print("4. Choc doux T=0,8 : E par jeu ; part des prompts où le jeton doux = jeton du témoin au pas t0+1")
rd = R("0.8", "doux")
for name, ix in (("écrits", hand), ("gabarits", tmpl)):
    c = ci(np.mean, win(rd[ix], EARLY)); print(f"   {name}: E_doux={win(rd[ix], EARLY).mean():+.4f} IC95[{c[0]:+.4f};{c[1]:+.4f}]")
td, tt = L("tok_0.8_doux"), L("tok_0.8_temoin")
print(f"   tok shape {td.shape} ; identiques au pas t0 : {np.mean(td[:, T0]==tt[:, T0]):.2f} ; identiques k=1–16 : {np.mean(td[:, T0+1:T0+17]==tt[:, T0+1:T0+17]):.2f}")
tf = L("tok_0.8_fort"); print(f"   fort : identiques k=1–16 : {np.mean(tf[:, T0+1:T0+17]==tt[:, T0+1:T0+17]):.2f}")

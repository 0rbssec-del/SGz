"""B1 après coup : plancher d'entropie, test des signes, positions où se trouve l'entropie."""
# Usage : python3 apres_coup.py <dossier contenant q/ (Qwen3) et s/ (SmolLM3), contenu des archives *_b1.zip>
import sys
import numpy as np
from scipy import stats

D = sys.argv[1]
for lab, sub in (("Qwen3-1.7B-Base", "q"), ("SmolLM3-3B-Base", "s")):
    L = lambda k: np.load(f"{D}/{sub}/{k}.npy", allow_pickle=False)
    SC, SW, SN = L("S_C").astype(float), L("S_W").astype(float), L("S_N").astype(float)
    LC, LW, LN = L("L_C"), L("L_W"), L("L_N")
    sc, sw = L("s_c"), L("s_w")
    h = lambda S, Lx: np.array([S[i, :Lx[i]].mean() for i in range(len(S))])
    HC, HW, HN = h(SC, LC), h(SW, LW), h(SN, LN)
    T = HW - HC
    print(f"\n=== {lab} : {len(T)} problèmes")
    print(f"H(C) : médiane {np.median(HC):.5f}, 90e centile {np.percentile(HC, 90):.5f} ; part des positions de la fenêtre où "
          f"S < 1e-3 : C {np.mean(np.concatenate([SC[i, :LC[i]] for i in range(len(SC))]) < 1e-3):.1%}")
    pos, neg, nul = np.mean(T > 0), np.mean(T < 0), np.mean(T == 0)
    st = stats.binomtest(int((T > 0).sum()), int((T != 0).sum()))
    w = stats.wilcoxon(T[T != 0])
    print(f"T > 0 : {pos:.1%}, T < 0 : {neg:.1%}, T = 0 : {nul:.1%} ; test des signes p = {st.pvalue:.3f} ; Wilcoxon p = {w.pvalue:.3f} ; "
          f"médiane de T {np.median(T):+.5f} ; moyenne {T.mean():+.5f}")
    tn = HN - HC
    print(f"témoin nul : T(nul) > 0 : {np.mean(tn > 0):.1%}, < 0 : {np.mean(tn < 0):.1%} ; signes p = "
          f"{stats.binomtest(int((tn > 0).sum()), int((tn != 0).sum())).pvalue:.3f}")
    m = np.minimum(LC, LW)
    d = np.array([[SW[i, k] - SC[i, k] if k < m[i] else np.nan for k in range(30)] for i in range(len(T))])
    prof = np.nanmean(d, 0)
    top = np.argsort(-np.nanmean(np.where(np.isnan(d), 0, SC[:, :30]), 0))[:5] + 1
    print(f"positions k où l'entropie après la bonne réponse est la plus haute : {top.tolist()} ; W − C à ces positions : "
          + ", ".join(f"k={k}: {prof[k - 1]:+.4f}" for k in top))
    print(f"entropie aux 3 premières positions (copie de la valeur) : C {SC[:, :3].mean():.5f}, W {SW[:, :3].mean():.5f}")

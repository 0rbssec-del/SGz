"""A1z : D₀ et I₀ selon le type de prompt, et IC naïf contre IC par grappes (thème, cadre de phrase)."""
# Usage : python3 strates.py <dossier contenant q/ (Qwen3) et s/ (SmolLM3), contenu des archives *_a1z.zip>
import sys
import numpy as np

D = sys.argv[1]
W = slice(2, 32)
rng = np.random.default_rng(1)
# Structure des 768 prompts (code A1c–A1z) : 0–63 écrits à la main (0–31 FR, 32–63 EN) ; puis TPL entrelacé FR, EN ;
# dans chaque langue : 24 thèmes courants puis 20 techniques, × 8 cadres de phrase.
n = 768
main = np.arange(n) < 64
lang = np.array(["fr" if (i < 32 or (i >= 64 and (i - 64) % 2 == 0)) else "en" for i in range(n)])
m = np.where(main, -1, (np.arange(n) - 64) // 2)
theme = np.where(main, -1, m // 8)
cadre = np.where(main, -1, m % 8)
tech = (~main) & (theme >= 24)
grappe_theme = np.where(main, np.arange(n), 1000 + theme * 2 + (lang == "en"))
grappe_cadre = np.where(main, np.arange(n), 5000 + cadre)


def ci_naif(x, B=2000):
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])


def ci_grappes(x, g, B=2000):
    u = np.unique(g)
    idx = {k: np.nonzero(g == k)[0] for k in u}
    out = []
    for _ in range(B):
        s = rng.choice(u, len(u))
        out.append(np.concatenate([x[idx[k]] for k in s]).mean())
    return np.percentile(out, [2.5, 97.5])


for lab, sub in (("Qwen3-1.7B-Base", "q"), ("SmolLM3-3B-Base", "s")):
    L = lambda k: np.load(f"{D}/{sub}/S_{k}.npy", allow_pickle=False)[:, W].mean(1).astype(float)
    St, Sr, Sc = L("temoin"), L("rejoue"), L("choc")
    D0, I0 = Sr - St, Sc - Sr
    print(f"\n=== {lab}")
    for nom, msk in (("tous", np.ones(n, bool)), ("écrits à la main", main), ("gabarits", ~main), ("français", lang == "fr"),
                     ("anglais", lang == "en"), ("gabarits courants", (~main) & ~tech), ("gabarits techniques", tech)):
        cd, cI = ci_naif(D0[msk]), ci_naif(I0[msk])
        print(f"{nom:<20} n={msk.sum():>3} : D₀ {D0[msk].mean():+.4f} [{cd[0]:+.4f} ; {cd[1]:+.4f}] ; I₀ {I0[msk].mean():+.4f} "
              f"[{cI[0]:+.4f} ; {cI[1]:+.4f}] ; I₀/D₀ {I0[msk].mean() / D0[msk].mean():+.2f}")
    par_cadre = [I0[cadre == c].mean() for c in range(8)]
    print(f"I₀ par cadre de phrase (8 cadres × 88 prompts) : min {min(par_cadre):+.4f}, max {max(par_cadre):+.4f}")
    for nom, x in (("D₀", D0), ("I₀", I0)):
        a, b, c = ci_naif(x), ci_grappes(x, grappe_theme), ci_grappes(x, grappe_cadre)
        print(f"largeur de l'IC95 de {nom} : naïf {b[1] - b[0] if False else a[1] - a[0]:.4f} ; par grappes de thème "
              f"{b[1] - b[0]:.4f} (×{(b[1] - b[0]) / (a[1] - a[0]):.2f}) ; par grappes de cadre {c[1] - c[0]:.4f} "
              f"(×{(c[1] - c[0]) / (a[1] - a[0]):.2f})")

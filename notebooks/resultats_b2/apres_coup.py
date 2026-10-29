"""B2 après coup : recouvrement entre modèles, langue, atomes (suites identiques), gain du préfixe sur le bassin."""
# Usage : python3 apres_coup.py <dossier contenant resume_b2.json et les dossiers <modèle>_T0.6/ (contenu des archives *_b2.zip)>
import json
import sys
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

D = sys.argv[1]
R = json.load(open(f"{D}/resume_b2.json"))
SEED = 20261029
mods = ["Qwen3-1.7B-Base", "SmolLM3-3B-Base"]
multi = {m: np.array(R[m]["T0.6"]["h"]["multimodal"]) for m in mods}
dist = {m: np.array(R[m]["T0.6"]["distinctes"]) for m in mods}
fr = np.arange(64) < 32
for m in mods:
    x = multi[m]
    print(f"{m} : multimodaux {x.sum()} ; français {x[fr].sum()}/32, anglais {x[~fr].sum()}/32 ; Fisher p = "
          f"{stats.fisher_exact([[x[fr].sum(), 32 - x[fr].sum()], [x[~fr].sum(), 32 - x[~fr].sum()]])[1]:.4f} ; "
          f"suites distinctes parmi les multimodaux : {np.round(dist[m][x], 2).tolist()}")
    ok = dist[m] >= 0.9
    print(f"   sans les prompts à moins de 90 % de suites distinctes ({(~ok).sum()} prompts) : {x[ok].sum()}/{ok.sum()} multimodaux "
          f"(π = {x[ok].mean():.2f})")
a, b = multi[mods[0]], multi[mods[1]]
k = int((a & b).sum())
p = stats.hypergeom.sf(k - 1, 64, int(a.sum()), int(b.sum()))
print(f"recouvrement : {k} prompts multimodaux pour les deux modèles (attendu au hasard {a.sum() * b.sum() / 64:.1f}) ; "
      f"p hypergéométrique = {p:.4f} ; prompts {np.nonzero(a & b)[0].tolist()}")
for m in mods:
    z = np.load(f"{D}/{m}_T0.6/H{R[m]['couches'][0]}_acp.npy", allow_pickle=False)
    TK = np.load(f"{D}/{m}_T0.6/TK.npy", allow_pickle=False)
    gains = []
    for p_ in np.nonzero(multi[m])[0]:
        lab = KMeans(2, n_init=10, random_state=int(SEED % 2 ** 31)).fit_predict(z[p_][:, :10].astype(np.float64))
        rng = np.random.default_rng(SEED + 7)
        best = (0.0, None)
        for kk in range(1, 33):
            cles = [tuple(r[:kk]) for r in TK[p_]]
            def pur(l):
                g = {}
                for c, v in zip(cles, l):
                    g.setdefault(c, []).append(v)
                return sum(max(np.bincount(v)) for v in g.values()) / len(l)
            gain = pur(lab) - np.mean([pur(rng.permutation(lab)) for _ in range(10)])
            if gain > best[0]:
                best = (gain, kk)
        gains.append((int(p_), round(best[0], 2), best[1], round(float(np.mean(lab)), 2)))
    print(f"{m} : gain maximal du préfixe sur le hasard (prompt, gain, k, part du bassin 1) : {gains}")

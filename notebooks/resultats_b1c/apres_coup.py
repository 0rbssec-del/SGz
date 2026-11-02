"""B1c après coup (non pré-enregistré) : AUC appariée par problème (l'erreur reçoit-elle un score de sonde plus haut que la
bonne réponse du même problème ?) et lien avec la taille de l'erreur.
Usage : python3 apres_coup.py <dossier contenant q/ (Qwen3) et s/ (SmolLM3), contenu des archives *_b1c.zip>"""
import glob
import sys

import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1]
src = open(__file__.replace("resultats_b1c/apres_coup.py", "sgz_savoir_b1c.py")).read()
src = src[:src.index("ITEMS_B1 = {")].replace('raise RuntimeError("Aucun GPU', 'None and RuntimeError("')
ns = {}
exec(compile(src, "b1c", "exec"), ns)                     # banque de problèmes de B1 (déterministe)
BONNE = ns["BONNE"]


def apparie(a, b):
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


for m, lab in (("q", "Qwen3-1.7B-Base"), ("s", "SmolLM3-3B-Base")):
    z = {f.split("/")[-1][:-4]: np.load(f, allow_pickle=False) for f in glob.glob(f"{D}/{m}/*.npy")}
    n = len(z["idx"])
    assert (z["y"][:n] == 0).all() and (z["y"][n:] == 1).all()
    o, osf = z["oof"], z["oof_surface"]
    C, W = BONNE[z["idx"]].astype(np.int64), z["w"].astype(np.int64)
    rng = np.random.default_rng(0)
    bs = [apparie(o[n:][ix], o[:n][ix]) for ix in (rng.integers(0, n, n) for _ in range(2000))]
    print(f"{lab} : AUC appariée sonde {apparie(o[n:], o[:n]):.3f} IC95 {np.percentile(bs, [2.5, 97.5]).round(3)} ; "
          f"surface {apparie(osf[n:], osf[:n]):.3f}")
    rel = np.abs(W - C) / C
    r, p = spearmanr(rel, o[n:] - o[:n])
    print(f"  Spearman(score erreur − score bonne réponse, erreur relative) = {r:+.3f} (p = {p:.3g})")

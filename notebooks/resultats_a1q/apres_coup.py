"""A1q, analyse après coup (descriptive, non pré-enregistrée) : effets absolus au pas suivant (k = 1) et en aval
(k = 2–31) pour chaque ensemble ablaté, afin de séparer « la recopie est cassée » (D_rep à k = 1 augmente, D_nouv
inchangé) de « le modèle est abîmé » (tous les effets s'écrasent). IC95 bootstrap sur les prompts (B = 1000).
Usage : python3 apres_coup.py <dossier npy Qwen3-1.7B-Base> <dossier npy Qwen3-1.7B>"""
import sys
import numpy as np

rng = np.random.default_rng(20261028)


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


for d in sys.argv[1:]:
    L = lambda k: np.load(f"{d}/{k}.npy").astype(float)
    v = np.load(f"{d}/valide.npy").astype(bool)
    E = {}
    for c in ("aucune", "induction", "motif", "dose_1%", "dose_6%"):
        ch, do, no = (L(f"S_{c}_{a}")[v] for a in ("choc1", "double", "nouveau"))
        E[c] = {"S0": ch[:, 2:32].mean(1), "rep": (do - ch)[:, 2:32].mean(1), "nouv": (no - ch)[:, 2:32].mean(1),
                "rep1": (do - ch)[:, 1], "nouv1": (no - ch)[:, 1]}
    P = int(v.sum())
    boots = [rng.integers(0, P, P) for _ in range(1000)]
    a = E["aucune"]
    print(f"\n{d} : {P} prompts")
    for c, e in E.items():
        sp = e["rep"].mean() / e["nouv"].mean()
        dsp = ci([e["rep"][ix].mean() / e["nouv"][ix].mean() - a["rep"][ix].mean() / a["nouv"][ix].mean() for ix in boots])
        dr1 = ci([e["rep1"][ix].mean() - a["rep1"][ix].mean() for ix in boots])
        dn1 = ci([e["nouv1"][ix].mean() - a["nouv1"][ix].mean() for ix in boots])
        print(f"  {c:<10} entropie de base {e['S0'].mean():.3f} | k = 1 : D_rep {e['rep1'].mean():+.3f} (écart à aucune IC95 "
              f"[{dr1[0]:+.3f} ; {dr1[1]:+.3f}]), D_nouv {e['nouv1'].mean():+.3f} ([{dn1[0]:+.3f} ; {dn1[1]:+.3f}]) | k = 2–31 : "
              f"D_rep {e['rep'].mean():+.4f}, D_nouv {e['nouv'].mean():+.4f}, Sp {sp:.3f} (ΔSp IC95 [{dsp[0]:+.3f} ; {dsp[1]:+.3f}])")

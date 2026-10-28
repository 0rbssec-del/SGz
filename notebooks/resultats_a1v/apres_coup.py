"""A1v, analyses après coup (non pré-enregistrées, descriptives). Entrée : dossiers des .npy de chaque modèle.
Usage : python3 -I apres_coup.py <dossier_qwen> <dossier_smollm>"""
import sys
import numpy as np

ZONE = (8, 151)
Z = slice(ZONE[0], ZONE[1] + 1)
rng = np.random.default_rng(0)


def contraste(x, w=8):
    out = np.full(x.shape, np.nan)
    for i in range(ZONE[0], ZONE[1] + 1):
        v = np.concatenate([x[:, max(0, i - w):i], x[:, i + 1:i + 1 + w]], axis=1)
        out[:, i] = x[:, i] - np.nanmedian(v, axis=1)
    return out


def auc(p, n):
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def ci_auc(p, n, B=1000):
    P = len(p)
    v = [auc(p[ix], n[ix]) for ix in (rng.integers(0, P, P) for _ in range(B))]
    return np.percentile(v, [2.5, 97.5])


for nom, d in zip(["Qwen3-1.7B-Base", "SmolLM3-3B-Base"], sys.argv[1:3]):
    L = lambda k: np.load(f"{d}/{k}.npy").astype(float)
    t0 = np.load(f"{d}/t0.npy")
    P = len(t0)
    rows = np.arange(P)
    c = {b: contraste(L(f"r1_{b}")) for b in ("temoin", "plausible", "rare")}
    s = {b: L(f"surp_{b}") for b in ("temoin", "plausible", "rare")}
    print(f"\n===== {nom} ({P} prompts)")
    a = auc(np.nanmax(c["rare"][:, Z], 1), np.nanmax(c["temoin"][:, Z], 1))
    print(f"[ANCRE] AUC max c1, rare, recalculée : {a:.3f} (log Kaggle : Qwen 0.528 / SmolLM3 0.551)")

    # A. Rang de t0 parmi les 144 positions de la zone, classées par c1 décroissant
    for b in ("plausible", "rare", "temoin"):
        x = c[b][:, Z]
        rk = (x > x[rows, t0 - ZONE[0]][:, None]).sum(1) + 1
        print(f"[RANG] {b:<9} rang de t0 selon c1 : médiane {np.median(rk):.0f} / 144 ; "
              f"top 1 {np.mean(rk == 1):.1%} ; top 5 {np.mean(rk <= 5):.1%} ; top 10 {np.mean(rk <= 10):.1%}")

    # B. Position connue, contraste local au lieu de r1 brut
    for b in ("plausible", "rare"):
        p, n = c[b][rows, t0], c["temoin"][rows, t0]
        lo, hi = ci_auc(p, n)
        print(f"[CONNUE] {b:<9} AUC c1(t0) injecté contre témoin : {auc(p, n):.3f} IC95 [{lo:.3f} ; {hi:.3f}]")

    # C. L'aiguille contre la plus haute paille : c1 au jeton inséré contre le max de c1 du témoin
    mt = np.nanmax(c["temoin"][:, Z], 1)
    for b in ("plausible", "rare"):
        ct = c[b][rows, t0]
        print(f"[PAILLE] {b:<9} c1 au jeton inséré : médiane {np.median(ct):.4f} ; max de c1 du témoin sur la zone : "
              f"médiane {np.median(mt):.4f} ; part des prompts où c1(t0) > max témoin : {np.mean(ct > mt):.1%}")

    # D. L'attention reçue suit-elle la surprise ? Témoin, toutes positions de la zone
    st, ctm = s["temoin"][:, Z].ravel(), c["temoin"][:, Z].ravel()
    from scipy.stats import spearmanr
    print(f"[SURPRISE] témoin, toutes positions : Spearman(surprise, c1) = {spearmanr(st, ctm)[0]:.3f}")
    # à surprise égale : jetons insérés plausibles contre jetons du témoin dans la même tranche de surprise
    sp, cp = s["plausible"][rows, t0], c["plausible"][rows, t0]
    for lo_, hi_ in ((5, 7), (7, 9), (9, 12)):
        mT, mP = (st >= lo_) & (st < hi_), (sp >= lo_) & (sp < hi_)
        if mT.sum() >= 20 and mP.sum() >= 20:
            print(f"[SURPRISE] tranche [{lo_} ; {hi_}[ nats : c1 médian témoin {np.median(ctm[mT]):.4f} (n = {mT.sum()}) ; "
                  f"inséré plausible {np.median(cp[mP]):.4f} (n = {mP.sum()}) ; AUC inséré contre témoin {auc(cp[mP], ctm[mT]):.3f}")
        else:
            print(f"[SURPRISE] tranche [{lo_} ; {hi_}[ nats : effectifs insuffisants (témoin {mT.sum()}, inséré {mP.sum()})")

    # E. Localisation du jeton plausible : surprise seule, puis départage par c1 parmi les 3 plus surprenants
    x, y = s["plausible"][:, Z], c["plausible"][:, Z]
    seul = np.mean(ZONE[0] + x.argmax(1) == t0)
    top3 = np.argsort(-x, 1)[:, :3]
    choix = top3[rows, np.nanargmax(np.take_along_axis(y, top3, 1), 1)]
    dans3 = np.mean((ZONE[0] + top3 == t0[:, None]).any(1))
    print(f"[LOC] plausible : surprise seule {seul:.1%} ; t0 parmi les 3 plus surprenants {dans3:.1%} ; "
          f"départage par c1 parmi ces 3 : {np.mean(ZONE[0] + choix == t0):.1%}")

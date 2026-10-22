# %% [markdown]
# # SGZ v3 — Bascule et longueur de contexte (analyse des mesures v2, CPU seul)
#
# **Entrée.** Les fichiers `*_mesures.npz` du run v2 (Qwen3 0.6B, 1.7B, 8B ; 64 prompts × 512 jetons × 12 températures).
# Sur Kaggle : « Add Input » → onglet *Your Work* / *Notebook Output* → le notebook v2. Aucun GPU ni Internet requis.
#
# **Contexte (v2, 22 oct. 2026).** χ croît avec la longueur L près de T ≈ 1,1 avec un exposant b ≈ 1,4–1,5, ce qui
# implique Var(surprisal cumulée) ∝ L^(b+1) ≈ L^2,45 > L² : impossible pour un processus stationnaire. Hypothèse :
# des séquences « basculent » en cours de génération (cohérent → bruit) à des moments différents. Et W4b (le bord
# monte de +0,06 à +0,08 entre début et fin) peut venir de la seule longueur de contexte.
#
# ## Volets et critères (fixés avant toute exécution ; bootstrap sur les prompts, B = 1000)
#
# **X0 — Non-stationnarité (descriptif chiffré).** a(T) = pente de log Var_prompts(Σ_{t<L} surprisal) contre log L,
# L ∈ {64, 128, 256, 512}. Un processus stationnaire donne a ≤ 2. « NON STATIONNAIRE » si l'IC95 de a(T*_χ) > 2.
#
# **X1 — Bascule.** Pour chaque séquence, statistique de rupture de moyenne vers le haut sur la surprisal :
# t_k = (moyenne après k − moyenne avant k) / (σ_poolée · √(1/k + 1/(L−k))), k ∈ [32, L−32] ; on garde max_k t_k et
# son k. Seuil = 95e centile de cette statistique sur les séquences témoins à T = 0,6 et T = 1,8 du même modèle.
# Température testée : T*_χ = argmax de χ₅₁₂ (calculée ici). Fraction f des séquences à T*_χ au-dessus du seuil.
# SOUTENU si f ≥ 0,30 et borne basse de l'IC95 > 0,10 ; RÉFUTÉ si borne haute < 0,15 ; sinon INDÉTERMINÉ.
# « Dates différentes » si l'écart interquartile des k des séquences qui basculent est ≥ 64 jetons (descriptif).
#
# **X2 — Témoin de longueur de contexte pour W4b.** Sur la base p_T (q = 1,45), T_c (centroïde de G1(T)) sur les jetons
# 0–127 et 384–511, pour les vraies générations et pour le témoin « trajectoires figées » (mêmes contextes générés à
# T = 1, seule la température change : toute dérive y est due à la position dans le contexte). Statistique :
# Δ = dérive réelle − dérive figée. « DÉRIVE PROPRE À LA DYNAMIQUE » si IC95(Δ) exclut 0 ;
# « EXPLIQUÉE PAR LA LONGUEUR DE CONTEXTE » si IC95(Δ) ⊂ [−0,03 ; 0,03] ; sinon INDÉTERMINÉ.
#
# **X3 — Individualité statique (descriptif).** Différence de T_c entre prompts français (0–31) et anglais (32–63),
# avec IC95 (les prompts diffèrent aussi par le contenu : descriptif seulement).
#
# **X4 — Ralentissement critique (affirmation A2 de la thèse « rétroaction négative », ajouté le 22 oct. 2026 avant
exécution).** Pour chaque T et chaque séquence, S₁,₄₅(p₁) le long de la trajectoire, tendance linéaire retirée ;
autocorrélation ρ(k) moyennée sur les prompts ; temps de retour τ(T) = premier retard (interpolé) où ρ < 1/e, plafonné
à 128 jetons. Témoin : même calcul après mélange de l'ordre des pas (τ attendu ≈ 0). SOUTENU si la fraction des
rééchantillonnages dont l'argmax de τ(T) tombe dans [1,0 ; 1,2] est ≥ 0,95 ; RÉFUTÉ si ≤ 0,05 ; sinon INDÉTERMINÉ.
ρ(1) par T est rapporté (descriptif). Risque connu d'avance : aux basses T, les boucles de répétition allongent τ
sans lien avec une criticité.

## Prédictions de l'assistant (écrites avant exécution) [spéculation]

- X4 : SOUTENU 35 %, RÉFUTÉ 40 % (boucles aux basses T), INDÉTERMINÉ 25 %.
#
# - X0 : NON STATIONNAIRE à T*_χ sur les 3 modèles (75 %).
# - X1 : SOUTENU 55 %, RÉFUTÉ 15 %, INDÉTERMINÉ 30 %.
# - X2 : expliquée par la longueur de contexte 40 %, propre à la dynamique 25 %, INDÉTERMINÉ 35 %.
#
# ## Limites connues
#
# - Le seuil de X1 suppose que T = 0,6 et T = 1,8 sont des régimes sans bascule vers le haut ; à T = 0,6, des boucles
#   peuvent apparaître (ruptures vers le bas, non comptées).
# - L'autocorrélation de la surprisal gonfle t_k ; c'est pourquoi le seuil est empirique (témoins du même modèle).
# - X2 compare deux bases (p_T réelle et p_T figée) qui partagent la même définition ; la base p₁ de W4b est rapportée
#   à côté, sans verdict.
#
# **À lire en priorité** : `[ENTRÉE]`, puis `[X0]` à `[X3]` et `[SYNTHÈSE]`.

# %%
import glob
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

MES_DIR = os.environ.get("MES_DIR", "/kaggle/input")
OUT = os.environ.get("OUT", "/kaggle/working/sgz_v3" if os.path.isdir("/kaggle/working") else "sgz_v3_out")
os.makedirs(OUT, exist_ok=True)
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261022
L_SIZES = [64, 128, 256, 512]
MIN_SEG = 32
Q = 1.45
rng = np.random.default_rng(SEED)

files = []
for root, dirs, names in os.walk(MES_DIR, followlinks=True):   # suit aussi les liens symboliques de /kaggle/input
    files += [os.path.join(root, n) for n in names if n.endswith("_mesures.npz")]
files = sorted(files)
print(f"[ENTRÉE] {len(files)} fichiers trouvés sous {MES_DIR} :")
for f in files:
    print(f"[ENTRÉE]   {f}")
if not files:
    print("[ENTRÉE] Contenu de l'entrée (3 niveaux), pour diagnostic :")
    for root, dirs, names in os.walk(MES_DIR, followlinks=True):
        depth = root[len(MES_DIR):].count(os.sep)
        if depth <= 3:
            print(f"[ENTRÉE]   {root}/ : {sorted(names)[:12]}")
    raise FileNotFoundError(
        "Aucun *_mesures.npz. Dans le panneau de droite : Input → Add Input → onglet « Your Work » → "
        "choisis le notebook v2 (sgz_criticite_v2) → « Add ». Puis relance Save & Run All.")

# %%
def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def centroid_T(curve, grid):
    w = np.asarray(curve) - np.min(curve)
    tr = getattr(np, "trapezoid", None) or np.trapz
    den = tr(w, grid)
    return float(tr(w * grid, grid) / den) if den > 0 else float("nan")


def g1_masked(S, mask):
    d = np.abs(np.diff(S, axis=-1))
    m = (mask[1:] & mask[:-1]).astype(float)
    return (d * m).sum(-1) / m.sum()


def rupture_haut(s, m=MIN_SEG):
    """max_k de t_k (moyenne après − moyenne avant, normalisée) et son k."""
    L = s.shape[0]
    cs, cs2 = np.cumsum(s), np.cumsum(s * s)
    k = np.arange(m, L - m + 1)
    n1, n2 = k, L - k
    s1, s2 = cs[k - 1], cs[-1] - cs[k - 1]
    q1, q2 = cs2[k - 1], cs2[-1] - cs2[k - 1]
    m1, m2 = s1 / n1, s2 / n2
    var = ((q1 - n1 * m1 ** 2) + (q2 - n2 * m2 ** 2)) / (L - 2)
    t = (m2 - m1) / np.sqrt(np.maximum(var, 1e-12) * (1 / n1 + 1 / n2))
    i = int(np.argmax(t))
    return float(t[i]), int(k[i])


def analyse_modele(path):
    z = np.load(path)
    grid = np.asarray(z["T_grid"], dtype=float)
    sur = np.asarray(z["surprisal"], dtype=float)          # (nT, P, L)
    nT, P, L = sur.shape
    Ls = [x for x in L_SIZES if x <= L]
    out = {"fichier": os.path.basename(path), "P": P, "L": L}
    boots = [rng.integers(0, P, P) for _ in range(BOOT_N)]

    chi = Ls[-1] * sur[:, :, :Ls[-1]].mean(2).var(1) / grid ** 2
    it = int(np.argmax(chi))
    out["T_star_chi"] = float(grid[it])

    # X0
    logL = np.log(Ls)

    def slope_a(i, ix):
        v = [sur[i][ix, :x].sum(1).var() for x in Ls]
        return np.polyfit(logL, np.log(np.asarray(v) + 1e-12), 1)[0]

    out["X0"] = {}
    for i, T in enumerate(grid):
        a = slope_a(i, np.arange(P))
        out["X0"][float(T)] = {"a": float(a)}
    a_b = [slope_a(it, ix) for ix in boots]
    lo, hi = ci(a_b)
    out["X0_verdict"] = {"a": out["X0"][float(grid[it])]["a"], "ci": (lo, hi),
                         "verdict": "NON STATIONNAIRE" if lo > 2 else ("COMPATIBLE STATIONNAIRE" if hi <= 2 else "INDÉTERMINÉ")}

    # X1
    stat = np.zeros((nT, P))
    kk = np.zeros((nT, P), dtype=int)
    for i in range(nT):
        for p in range(P):
            stat[i, p], kk[i, p] = rupture_haut(sur[i, p])
    ctrl = [int(np.argmin(np.abs(grid - 0.6))), int(np.argmin(np.abs(grid - 1.8)))]
    thr = float(np.percentile(stat[ctrl].ravel(), 95))
    above = stat > thr
    f = float(above[it].mean())
    fb = [above[it][ix].mean() for ix in boots]
    flo, fhi = ci(fb)
    v = "SOUTENU" if (f >= 0.30 and flo > 0.10) else ("RÉFUTÉ" if fhi < 0.15 else "INDÉTERMINÉ")
    ks = kk[it][above[it]]
    iqr = float(np.subtract(*np.percentile(ks, [75, 25]))) if ks.size >= 4 else float("nan")
    out["X1"] = {"seuil": thr, "fraction": f, "ci": (flo, fhi), "verdict": v,
                 "fraction_par_T": {float(T): float(above[i].mean()) for i, T in enumerate(grid)},
                 "k_median": float(np.median(ks)) if ks.size else float("nan"), "k_iqr": iqr,
                 "dates_differentes": bool(iqr >= 64) if np.isfinite(iqr) else None}
    # ampleur du saut à T*_χ pour les séquences qui basculent
    jumps = []
    for p in np.where(above[it])[0]:
        k0 = kk[it, p]
        jumps.append(sur[it, p, k0:].mean() - sur[it, p, :k0].mean())
    out["X1"]["saut_moyen_nats"] = float(np.mean(jumps)) if jumps else float("nan")

    # X2
    Sr, Sf = np.asarray(z[f"S_pT_q{Q}"], dtype=float), np.asarray(z[f"S_frozen_q{Q}"], dtype=float)
    Sp1 = np.asarray(z[f"S_p1_q{Q}"], dtype=float)
    pos = np.arange(L)
    q4 = L // 4
    early, late = pos < q4, pos >= L - q4

    def tc(S, mask):
        g = np.stack([g1_masked(S[i], mask) for i in range(nT)])      # (nT, P)
        return np.array([centroid_T(g[:, p], grid) for p in range(P)])

    dr = tc(Sr, late) - tc(Sr, early)
    df = tc(Sf, late) - tc(Sf, early)
    dp1 = tc(Sp1, late) - tc(Sp1, early)
    delta = dr - df
    db = [np.nanmean(delta[ix]) for ix in boots]
    dlo, dhi = ci(db)
    v2 = ("DÉRIVE PROPRE À LA DYNAMIQUE" if (dlo > 0 or dhi < 0)
          else "EXPLIQUÉE PAR LA LONGUEUR DE CONTEXTE" if (dlo >= -0.03 and dhi <= 0.03) else "INDÉTERMINÉ")
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    out["X2"] = {"derive_reelle_pT": float(np.nanmean(dr)), "derive_figee_pT": float(np.nanmean(df)),
                 "derive_p1_W4b": float(np.nanmean(dp1)), "delta": float(np.nanmean(delta)), "ci": (dlo, dhi),
                 "verdict": v2,
                 "S_p1_T1_debut": float(Sp1[i1][:, early].mean()), "S_p1_T1_fin": float(Sp1[i1][:, late].mean())}

    # X3
    tall = tc(Sp1, np.ones(L, dtype=bool))
    h = P // 2
    dfe = float(np.nanmean(tall[:h]) - np.nanmean(tall[h:]))
    bb = [np.nanmean(tall[:h][rng.integers(0, h, h)]) - np.nanmean(tall[h:][rng.integers(0, P - h, P - h)]) for _ in range(BOOT_N)]
    out["X3"] = {"Tc_francais": float(np.nanmean(tall[:h])), "Tc_anglais": float(np.nanmean(tall[h:])),
                 "difference": dfe, "ci": ci(bb)}
    # X4
    S4 = Sp1
    lags = np.arange(0, 129)

    def tau_curve(Sm):
        x = Sm - np.polyval(np.polyfit(pos, Sm.T, 1), pos[:, None]).T     # tendance linéaire retirée par séquence
        x = x - x.mean(1, keepdims=True)
        F = np.fft.rfft(x, n=2 * L, axis=1)
        ac = np.fft.irfft(F * np.conj(F), axis=1)[:, :L]
        v0 = ac[:, :1]
        ac = np.where(v0 > 1e-20, ac / np.where(v0 > 1e-20, v0, 1.0), np.nan)   # séquence constante : exclue
        return ac                                                            # (P, L)

    def tau_from(rho):
        if not np.isfinite(rho[0]):
            return float("nan")
        below = np.where(rho[1:129] < 1 / np.e)[0]
        if below.size == 0:
            return 128.0
        k = below[0] + 1
        r0, r1 = rho[k - 1], rho[k]
        return float(k - 1 + (r0 - 1 / np.e) / (r0 - r1)) if r0 != r1 else float(k)

    AC = np.stack([tau_curve(S4[i]) for i in range(nT)])                      # (nT, P, L)
    perm = np.argsort(np.random.default_rng(SEED + 4).random((P, L)), axis=1)
    ACm = np.stack([tau_curve(np.take_along_axis(S4[i], perm, axis=1)) for i in range(nT)])
    tau = np.array([tau_from(np.nanmean(AC[i], 0)) for i in range(nT)])
    tau_m = np.array([tau_from(np.nanmean(ACm[i], 0)) for i in range(nT)])
    stars = np.array([grid[int(np.nanargmax([tau_from(np.nanmean(AC[i][ix], 0)) for i in range(nT)]))] for ix in boots])
    fr = float(np.mean((stars >= 1.0 - 1e-9) & (stars <= 1.2 + 1e-9)))
    out["X4"] = {"tau": tau.tolist(), "tau_melange": tau_m.tolist(), "rho1": [float(np.nanmean(AC[i], 0)[1]) for i in range(nT)],
                 "T_star_tau": float(grid[int(np.nanargmax(tau))]), "frac_fenetre": fr, "ci_T_star": ci(stars),
                 "verdict": "SOUTENU" if fr >= 0.95 else ("RÉFUTÉ" if fr <= 0.05 else "INDÉTERMINÉ")}
    return out

# %%
RES = {}
for f in files:
    label = os.path.basename(f).replace("_mesures.npz", "")
    r = analyse_modele(f)
    RES[label] = r
    print(f"\n{'=' * 90}\n[MODÈLE] {label} : {r['P']} prompts × {r['L']} jetons ; T*_χ = {r['T_star_chi']}")
    x0 = r["X0_verdict"]
    print(f"[X0] {label} : {x0['verdict']} — a(T*_χ) = {x0['a']:.3f}, IC95 [{x0['ci'][0]:.3f} ; {x0['ci'][1]:.3f}] (stationnaire : a ≤ 2)")
    print("[X0] a(T) : " + " ; ".join(f"T={T}: {d['a']:.2f}" for T, d in r["X0"].items()))
    x1 = r["X1"]
    print(f"[X1] {label} : {x1['verdict']} — fraction à T*_χ {x1['fraction']:.3f}, IC95 [{x1['ci'][0]:.3f} ; {x1['ci'][1]:.3f}] "
          f"(seuil {x1['seuil']:.2f} = 95e centile des témoins T=0,6 et 1,8)")
    print(f"[X1] dates de bascule : médiane {x1['k_median']:.0f}, écart interquartile {x1['k_iqr']:.0f} jetons ; "
          f"dates différentes : {x1['dates_differentes']} ; saut moyen de surprisal {x1['saut_moyen_nats']:+.2f} nats")
    print("[X1] fraction au-dessus du seuil par T : " + " ; ".join(f"{T}: {v:.2f}" for T, v in x1["fraction_par_T"].items()))
    x2 = r["X2"]
    print(f"[X2] {label} : {x2['verdict']} — dérive de T_c (fin − début) : réelle pT {x2['derive_reelle_pT']:+.3f}, "
          f"figée {x2['derive_figee_pT']:+.3f}, Δ = {x2['delta']:+.3f} IC95 [{x2['ci'][0]:+.3f} ; {x2['ci'][1]:+.3f}] ; "
          f"base p1 (W4b) {x2['derive_p1_W4b']:+.3f}")
    print(f"[X2] S_1,45(p1) à T=1 : début {x2['S_p1_T1_debut']:.3f}, fin {x2['S_p1_T1_fin']:.3f}")
    x3 = r["X3"]
    print(f"[X3] {label} : T_c français {x3['Tc_francais']:.3f}, anglais {x3['Tc_anglais']:.3f}, "
          f"différence {x3['difference']:+.3f} IC95 [{x3['ci'][0]:+.3f} ; {x3['ci'][1]:+.3f}]")
    x4 = r["X4"]
    print(f"[X4] {label} : {x4['verdict']} — T* de τ = {x4['T_star_tau']}, fraction dans [1,0 ; 1,2] {x4['frac_fenetre']:.3f}, "
          f"IC95 de T* [{x4['ci_T_star'][0]} ; {x4['ci_T_star'][1]}]")
    print("[X4] τ(T) en jetons : " + " ; ".join(f"{T}: {v:.1f}" for T, v in zip(r['X0'].keys(), x4['tau'])))
    print("[X4] τ mélangé      : " + " ; ".join(f"{T}: {v:.1f}" for T, v in zip(r['X0'].keys(), x4['tau_melange'])))
    print("[X4] ρ(1)           : " + " ; ".join(f"{T}: {v:.2f}" for T, v in zip(r['X0'].keys(), x4['rho1'])))

print(f"\n{'=' * 90}")
for label, r in RES.items():
    print(f"[SYNTHÈSE] {label:<18} X0 : {r['X0_verdict']['verdict']:<24} X1 : {r['X1']['verdict']:<12} X2 : {r['X2']['verdict']:<38} X4 : {r['X4']['verdict']}")
with open(os.path.join(OUT, "resume_v3.json"), "w") as fh:
    json.dump(RES, fh, ensure_ascii=False, indent=1, default=str)
print(f"[SORTIE] {os.path.join(OUT, 'resume_v3.json')}")

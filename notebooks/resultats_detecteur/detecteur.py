"""Détecteur d'anomalies par la seule trace d'entropie (analyse CPU des données A1p et A1o essai 2, sans nouveau calcul GPU).

Question : un moniteur qui ne voit que l'incertitude du modèle (entropie de Tsallis q = 1,45 à chaque pas, celle que les
notebooks enregistrent) remarque-t-il qu'un jeton anormal (rang 1000–5000) a été inséré ? La surprise du jeton lui-même
n'est pas utilisée : quand le jeton est visible, elle le trahit presque toujours (10 à 48 nats contre ~1 nat pour un jeton
tiré à T = 0,6) [hypothèse forte, non calculée ici faute de surprise enregistrée pour chaque jeton].

Données : A1p (Qwen3-1.7B-Base et Qwen3-1.7B ; 768 prompts ; génération libre de 128 jetons ; choc à t = 64, donc k = 1 à
l'indice 65) et A1o essai 2 (Qwen3-4B, 384 conversations × 6 tours de 48 jetons, formats brut et gabarit ; choc au tour 1,
position 8, k = 1 à l'indice 9). Chaque prompt donne une trace témoin (négative) et une trace choquée (positive).

Critères (fixés avant tout calcul, commit séparé) ; AUC de Mann-Whitney, IC95 par bootstrap sur les prompts (les deux traces
d'un prompt tirées ensemble, B = 1000) :
- D0 (position connue, k = 1 seul, borne haute d'un détecteur à un pas) : score = S[t + 1].
- D1 (position inconnue, détecteur de pic) : score = max de S sur les indices ≥ 8 de la génération (A1p) ou du tour 1 (A1o).
- D2 (en aval, position connue, le jeton et le pas suivant masqués) : score = moyenne de S sur k = 2–31 (A1p) ou 2–31 du tour 1 (A1o).
- D4 (tours suivants, A1o seulement) : score = moyenne de S sur les tours 2 à 6.
  Règle commune : DÉTECTABLE si borne basse ≥ 0,75 ; FAIBLE si borne basse > 0,5 ; INVERSE si borne haute < 0,5 ;
  sinon NON DÉTECTABLE.
- L1 (localisation, traces choquées) : position prédite = argmax de S (mêmes indices que D1) ; exacte si = t + 1.
  LOCALISABLE si borne basse de la part exacte > 0,5 ; PARTIELLE si borne basse > 5 × le hasard (1 / nombre d'indices) ;
  sinon NON LOCALISABLE.
- Descriptifs : saut maximal S[i + 1] − S[i] comme autre détecteur de pic ; localisation à ±1 près ; entropie moyenne de
  la trace entière (détecteur naïf).

Prédictions de l'assistant (écrites avant calcul) [spéculation] : D0 DÉTECTABLE 60 % ; D1 FAIBLE 55 %, DÉTECTABLE 20 % ;
D2 FAIBLE 60 % ; L1 PARTIELLE 50 %, LOCALISABLE 15 % ; D4 brut FAIBLE 45 %, D4 gabarit NON DÉTECTABLE 70 %.

Usage : python3 detecteur.py <A1p base> <A1p instruct> <A1o brut> <A1o gabarit> (dossiers de .npy dézippés)
"""
import sys
import numpy as np
from scipy.stats import rankdata

B, SEED, DEBUT = 1000, 20261028, 8


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def regle_auc(c):
    if c[0] >= 0.75:
        return "DÉTECTABLE"
    if c[0] > 0.5:
        return "FAIBLE"
    if c[1] < 0.5:
        return "INVERSE"
    return "NON DÉTECTABLE"


def juger_auc(nom, sp, sn, boots):
    a = auc(sp, sn)
    c = ci([auc(sp[ix], sn[ix]) for ix in boots])
    v = regle_auc(c)
    print(f"  {nom:<44} AUC {a:.3f} IC95 [{c[0]:.3f} ; {c[1]:.3f}] → {v}")
    return v


def localiser(Sc, k1, boots):
    pred = DEBUT + np.argmax(Sc[:, DEBUT:], axis=1)
    ex, pr = (pred == k1).astype(float), (np.abs(pred - k1) <= 1).astype(float)
    c = ci([ex[ix].mean() for ix in boots])
    hasard = 1 / (Sc.shape[1] - DEBUT)
    v = "LOCALISABLE" if c[0] > 0.5 else ("PARTIELLE" if c[0] > 5 * hasard else "NON LOCALISABLE")
    print(f"  L1 localisation par le pic d'entropie : exacte {ex.mean():.1%} IC95 [{c[0]:.1%} ; {c[1]:.1%}] (hasard "
          f"{hasard:.1%}) ; à ±1 près {pr.mean():.1%} → {v}")
    return v


def saut_max(S):
    return np.diff(S[:, DEBUT:], axis=1).max(1)


rng = np.random.default_rng(SEED)
pbase, pinst, obrut, ogab = sys.argv[1:5]
print("=== A1p : génération libre, choc à t = 64 ===")
for nom, d in (("Qwen3-1.7B-Base", pbase), ("Qwen3-1.7B", pinst)):
    St, Sc = (np.load(f"{d}/{k}.npy").astype(float) for k in ("S_temoin", "S_choc1"))
    P, k1 = len(St), 65
    boots = [rng.integers(0, P, P) for _ in range(B)]
    print(f"{nom} ({P} prompts)")
    juger_auc("D0 k = 1 seul (position connue)", Sc[:, k1], St[:, k1], boots)
    juger_auc("D1 pic d'entropie (position inconnue)", Sc[:, DEBUT:].max(1), St[:, DEBUT:].max(1), boots)
    juger_auc("D2 en aval, k = 2–31 (position connue)", Sc[:, k1 + 1:k1 + 31].mean(1), St[:, k1 + 1:k1 + 31].mean(1), boots)
    juger_auc("(descriptif) saut maximal", saut_max(Sc), saut_max(St), boots)
    juger_auc("(descriptif) entropie moyenne de la trace", Sc[:, DEBUT:].mean(1), St[:, DEBUT:].mean(1), boots)
    localiser(Sc, k1, boots)
print("\n=== A1o essai 2 : Qwen3-4B, 6 tours, choc au tour 1 ===")
for nom, d in (("brut", obrut), ("gabarit", ogab)):
    St, Sc = (np.load(f"{d}/{k}_t1.npy").astype(float) for k in ("S_temoin", "S_choc1"))
    Lt = np.concatenate([np.load(f"{d}/S_temoin_t{t}.npy") for t in range(2, 7)], 1).astype(float)
    Lc = np.concatenate([np.load(f"{d}/S_choc1_t{t}.npy") for t in range(2, 7)], 1).astype(float)
    P, k1 = len(St), 9
    boots = [rng.integers(0, P, P) for _ in range(B)]
    print(f"{nom} ({P} conversations)")
    juger_auc("D0 k = 1 seul (position connue)", Sc[:, k1], St[:, k1], boots)
    juger_auc("D1 pic d'entropie au tour 1 (position inconnue)", Sc[:, DEBUT:].max(1), St[:, DEBUT:].max(1), boots)
    juger_auc("D2 en aval, k = 2–31 du tour 1", Sc[:, k1 + 1:k1 + 31].mean(1), St[:, k1 + 1:k1 + 31].mean(1), boots)
    juger_auc("D4 tours 2 à 6", Lc.mean(1), Lt.mean(1), boots)
    localiser(Sc, k1, boots)

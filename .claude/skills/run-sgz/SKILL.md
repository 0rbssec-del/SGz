---
name: run-sgz
description: Lancer les scripts d'expérience du dépôt SGz (attracteur de Lorenz, AETHER-SIEVE et son audit) depuis un conteneur Linux neuf, CPU seul, et vérifier leur sortie.
---

# Lancer les expériences SGz

Dépôt de scripts Python autonomes (pas de paquet, pas de tests). Chaque script
s'exécute depuis la racine du dépôt et écrit ses sorties dans le dossier courant.

## 1. Dépendances

Le conteneur de base n'a ni `numba`, ni `scipy`, ni `matplotlib` :

```bash
pip install -r requirements.txt
```

Vérification : `python3 -c "import numpy, scipy, numba, matplotlib"`.

## 2. Scripts, durée (4 cœurs CPU) et sortie attendue

| Commande | Durée | Sortie attendue | Fichiers écrits |
|---|---|---|---|
| `python3 lorenz_stretching.py` | ~9 s | lignes `Spearman(...)`, puis `Figure écrite : lorenz_stretching.png` | `lorenz_stretching.png` |
| `python3 lorenz_stretching.py 0.5` | ~8 s | idem, fenêtre FTLE T = 0.5 | `lorenz_stretching_T0.5.png` |
| `python3 lorenz_ftle_sweep.py` | ~50 s | contrastes `T=… vs T=… : … (t=…, 10/10 CI positives)` | `lorenz_ftle_sweep.png`, `lorenz_ftle_sweep.csv` |
| `python3 aether_sieve_fixed.py` | ~10 s | tableau « AETHER-SIEVE v1.0 », corrélation brut ≈ 0.558 → purifié ≈ 0.849 | aucun |
| `python3 aether_sieve_audit.py` | ~55 s | blocs [C1]…[C4] ; `corr(organique, purifié moteur) 0.8345 ± 0.0196` | aucun (rediriger vers `aether_sieve_audit.txt` pour mettre à jour le journal) |

Toutes les graines sont fixées : les chiffres doivent être identiques d'une exécution
à l'autre. `lorenz_ftle_sweep.csv` régénéré est identique octet pour octet à la version commitée.

## 3. Pièges connus

- `python3 aether_sieve.py` (code d'origine) **échoue volontairement** à la compilation
  Numba : `AssertionError: unexpected cycle in lookup()` (deux `if` imbriqués dans un
  `prange`, Numba 0.61 et 0.68). Utiliser `aether_sieve_fixed.py`, dont l'audit dépend.
- Le premier lancement d'un script Numba compile les noyaux (quelques secondes) ;
  `cache=True` écrit dans `__pycache__/` (ignoré par git).
- `RuntimeWarning: divide by zero` sur `pink_env` dans les scripts AETHER-SIEVE : bénin
  (le `np.where` met la fréquence nulle à 0).
- Les scripts Lorenz **écrasent** les PNG/CSV commités. Pour une simple vérification,
  copier les `.py` dans un dossier temporaire et lancer depuis là.
- Matplotlib est forcé en backend `Agg` : pas d'affichage, uniquement des fichiers PNG.
  Ouvrir le PNG produit (outil Read) pour vérifier la figure.

---
name: run-sgz
description: Lancer les scripts d'expérience du dépôt SGz (attracteur de Lorenz : étirement local, balayage FTLE) depuis un conteneur Linux neuf, CPU seul, et vérifier leur sortie.
---

# Lancer les expériences SGz

Dépôt de scripts Python autonomes (pas de paquet, pas de tests). Chaque script
s'exécute depuis la racine du dépôt et écrit ses sorties dans le dossier courant.

## 1. Dépendances

Le conteneur de base n'a pas `matplotlib` :

```bash
pip install -r requirements.txt
```

Vérification : `python3 -c "import numpy, matplotlib"`.

## 2. Scripts, durée (4 cœurs CPU) et sortie attendue

| Commande | Durée | Sortie attendue | Fichiers écrits |
|---|---|---|---|
| `python3 lorenz_stretching.py` | ~9 s | lignes `Spearman(...)`, puis `Figure écrite : lorenz_stretching.png` | `lorenz_stretching.png` |
| `python3 lorenz_stretching.py 0.5` | ~8 s | idem, fenêtre FTLE T = 0.5 | `lorenz_stretching_T0.5.png` |
| `python3 lorenz_ftle_sweep.py` | ~50 s | contrastes `T=… vs T=… : … (t=…, 10/10 CI positives)` | `lorenz_ftle_sweep.png`, `lorenz_ftle_sweep.csv` |

Toutes les graines sont fixées : `lorenz_ftle_sweep.csv` régénéré est identique octet
pour octet à la version commitée.

## 3. Pièges connus

- Les scripts **écrasent** les PNG/CSV commités. Pour une simple vérification,
  copier les `.py` dans un dossier temporaire et lancer depuis là.
- Matplotlib est forcé en backend `Agg` : pas d'affichage, uniquement des fichiers PNG.
  Ouvrir le PNG produit (outil Read) pour vérifier la figure.
- AETHER-SIEVE n'est plus dans ce dépôt : il a son propre projet (`aether-sieve`).

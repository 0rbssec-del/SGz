# Analyse de `critical-asymmetry-llm` (2026-10-12)

## Conclusion

En l'état d'origine, le dépôt ne peut pas tester sa propre thèse :
- il ne s'exécute pas ;
- la mesure A vaut environ 0 ;
- la boucle PID ne contrôle pas ce qu'elle mesure ;
- aucun code ne produit les résultats de `docs/paper.md` ;
- `docs/proofs.md` ne contient que des énoncés.

## Bugs vérifiés par exécution

| # | Fichier | Problème | Statut |
|---|---|---|---|
| 1 | `src/agents/caa_agent.py:13` | Deux instructions sur une ligne : `SyntaxError`, la suite de tests ne se charge pas | **corrigé** |
| 2 | `metrics.py`, `multimodal_metrics.py` | `F.kl_div(..., "batchmean")` sur des logits 1D divise par la taille du vocabulaire (KL vraie 8.30, φ2 calculé 0.00026) | **corrigé** (`kl_logits`, KL sommée sur le vocabulaire) |
| 3 | agents, `llm_wrapper.py`, `metrics.py` | `device="cuda"` par défaut : plante sans GPU | **corrigé** (`cpu` par défaut ; `configs/default.yaml` inchangé) |
| 4 | `caa_agent.py`, `sampler.py` | Boucle ouverte : A est mesuré avant l'échantillonnage, donc (T, p) n'agit pas sur A ; T ne change jamais ; l'intégrale s'emballe | non corrigé (problème de conception) |
| 5 | `caa_agent.py` | Le méta-contrôleur calcule un A* que le PID n'utilise jamais | non corrigé |
| 6 | `caa_agent.py`, `multimodal_agent.py` | Avec `model=None`, A ~ U(0.4, 0.6) : taux critique de 1.0 par construction | non corrigé |
| 7 | `multimodal_agent.py` | φ1–φ6 fixés à 0.5, donc A_MM ∈ [0.31, 0.69] quoi qu'il arrive ; « logits » = états cachés moyennés | non corrigé |
| 8 | `docs/proofs.md` | Théorème 4 : α > ‖W‖₂ violé dans 100 % des tirages par défaut (‖W‖₂ médiane 1.88, α = 0.1) | — |
| 9 | `benchmarks/evaluation.py` | Scoreurs factices (`0.5`, `hash() % 100`) | — |

Tests : 0 exécutés à l'origine (erreur de collecte) ; 21/21 passent après correction, dont un test de non-régression pour le bug 2.

Effet secondaire de la correction 2 : `tanh(KL)` peut saturer. Sur des logits aléatoires, KL = 8.3 donne φ2 ≈ 1.0 (vérifié). Sur un vrai modèle, l'ampleur de la KL n'est pas vérifiée ; le script d'expérience enregistre donc la KL brute en plus de `tanh`.

## Expérience discriminante : `experiments/phi6_vs_sampling.py`

**Question.** La commande (T, top-p) peut-elle déplacer φ6, quand φ6 est calculé sur la continuation générée (boucle fermée) ?

**Plan.**
- Grille : T ∈ {0.5, 0.8, 1.0, 1.3, 1.6}, top-p ∈ {0.5, 0.8, 0.95, 0.99}.
- 12 paires (info privée, prompt), 3 graines, 30 tokens.
- Contrôle : φ6 statique (version du dépôt), constant par construction.

**Critère de réfutation (fixé avant exécution).** Si l'écart de φ6_gen entre les cellules extrêmes est ≤ 2 écarts-types entre graines, alors (T, p) ne pilote pas A.

**Statut.** Pas exécuté sur GPT-2 : `huggingface.co` est bloqué par le réseau de l'environnement cloud. Seul le test de fumée (modèle aléatoire, `--smoke`) a tourné ; il valide le code, pas la thèse.

    PYTHONPATH=. python experiments/phi6_vs_sampling.py --model gpt2   # ~10 min sur CPU 4 cœurs (estimation)

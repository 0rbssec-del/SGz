# A2 — addendum au pré-enregistrement (24 oct. 2026, avant tout résultat A2)

Question ajoutée : la thèse d'Erick « le modèle se calibre pour atteindre une position x » prédit un **point fixe** :
après adaptation, le niveau d'entropie ne dépend plus de la dose de corruption.

Règle de lecture (fixée avant d'avoir vu les résultats d'A2 ; calcul sur le fichier `Qwen3-1.7B-Base_a2.npz`,
bootstrap apparié sur les 256 prompts, B = 1000) :

- Δ = S̄(H2, dose 0,75) − S̄(H2, dose 0), avec H2 = pas 256–447 et S = S₁,₄₅(p₁).
- POINT FIXE si IC95(Δ) ⊂ [−0,05 ; +0,05].
- PAS DE POINT FIXE si la borne basse de IC95(Δ) dépasse +0,05.
- Sinon INDÉTERMINÉ.

Prédiction de l'assistant [spéculation] : PAS DE POINT FIXE 75 %, INDÉTERMINÉ 15 %, POINT FIXE 10 %.

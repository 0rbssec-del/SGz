# Veille frontière — excès de valeur (‖W_O v‖) d'un jeton anormal et amplification de l'entropie

Date : 27 octobre 2026. Question : existe-t-il un test causal, sur un LLM décodeur récent, d'un jeton
improbable injecté dans le contexte dont l'excès de norme de valeur prédirait une amplification de
l'entropie dans la génération qui suit ? Fait suite à la recherche économe du 27 oct. (voir doc de suivi),
qui avait établi l'antériorité de la méthode (Kobayashi et al. 2020) sans tester la littérature récente de
façon systématique.

## Couches exécutées

1. **Récence (90 jours)** : ✓. WebSearch anglais + alphaXiv (`prioritize: recency`).
2. **Graphe de citations** : ✓ partiel. Pivots : Kobayashi et al. 2020 (203 citants, via Scite, fait le 27 oct.
   lors de la recherche économe) ; Yan et al. 2024 (3 citants, via Scite, faible couverture — `low_coverage_seeds`).
   Semantic Scholar et OpenAlex inaccessibles (WebFetch : `ENOTFOUND` sur `api.semanticscholar.org`,
   `api.openalex.org` et même `arxiv.org` — panne réseau de l'outil cette session, pas un blocage ciblé),
   compensé par Scite seul.
3. **Revues de pairs (OpenReview)** : ✓ partiel. Recherche par mots-clés ; rien de directement pertinent
   trouvé sur OpenReview précisément sur l'excès de valeur causal.
4. **Négatif, errata, transformer-circuits.pub, Alignment Forum** : ✗. Le filtre `site:` n'a pas été respecté
   par l'outil de recherche pour `transformer-circuits.pub` ; WebFetch indisponible (panne, voir couche 2).
   Recherche de réplication/critique de Kobayashi et al. par mots-clés : rien trouvé (silence, pas une
   réfutation confirmée).
5. **Domaines voisins et langues** : ✓. Français, chinois, allemand, japonais interrogés avec des requêtes
   écrites dans chacune de ces langues. **Dans les quatre, seulement des reprises de travaux anglophones
   (Kobayashi 2020, Guo et al. 2024, VATP) ont été trouvées : aucune étude primaire dans une autre langue
   sur cette question précise.**
6. **Code et artefacts** : ✓ partiel. Un dépôt trouvé (SirLLM, tangentiel) ; rien qui corresponde
   directement au sujet.
7. **Éditeurs, .gov, .edu** : ✓ partiel. ACL Anthology interrogée (2025–2026) : rien qui combine norme de
   valeur, surprise et entropie. IEEE Xplore et dblp.org non interrogés spécifiquement (hors budget).

Coût réel : 20 recherches (haut de la fourchette budgétée).

## Ce qui change par rapport à la recherche économe du 27 oct.

La recherche économe avait trouvé Kobayashi et al. (2020, sur BERT) et l'image miroir des puits
d'attention. Cette veille ajoute :
- **une extension directe à un LLM décodeur moderne** (Llama 3-8B) du mécanisme « valeur vidée » ;
- **un mécanisme concurrent et complémentaire** : des neurones dédiés qui régulent l'entropie de sortie,
  indépendamment de la norme des valeurs dans l'attention ;
- **une preuve théorique et empirique d'atténuation**, sur une injection différente (préfixe continu, pas un
  jeton discret forcé en cours de génération), qui va dans le sens inverse de notre verdict AMPLIFICATION ;
- **confirmation de l'absence** : aucune étude trouvée, dans aucune des couches ni des 4 langues, ne teste
  exactement notre protocole (injection causale d'un jeton discret + excès de valeur + lien avec
  l'amplification de la trajectoire qui suit).

## Table datée (plus récent en premier)

| Date | Travail | Résultat chiffré | Méthode | Statut de lecture |
|---|---|---|---|---|
| 2026-10 (prépub.) | GlitchPatch (USTC, UW, Wuhan) | Répare les jetons « glitch » par retokenisation locale | Jetons anormaux du vocabulaire, pas d'injection causale avec mesure de norme | Résumé |
| 2026-05 (prépub.) | Kim, Lee et al., *Random Soft Prompts* | Entropie en hausse en début de génération puis retour à la ligne de base ; borne théorique : la masse d'attention sur l'injection → 0 quand le cache KV grandit | Injection d'un préfixe continu aléatoire (10–20 vecteurs), 3 benchmarks de raisonnement, LLaMA-3.1-8B, Qwen2.5-Math | Lu en entier |
| 2024-10 (EMNLP Findings) | VATP, *Attention Score is not All You Need* | La norme des valeurs varie fortement par jeton (y compris chez des modèles Llama) ; score = attention × norme L1 bat le score d'attention seul sur 12/16 tâches LongBench | KV-cache, pas de test causal sur un jeton injecté | Résumé (WebSearch) |
| 2024-10 (NeurIPS) | Stolfo, Wu et al., *Confidence Regulation Neurons* | Neurones « d'entropie » à grande norme de poids, effet médié par LayerNorm ; neurones « de fréquence des jetons » qui déplacent les logits selon log(fréquence) | Médiation causale, SVD de la matrice de dé-plongement, 7 modèles (GPT-2 à LLaMA2-7B) | Lu en entier (rapport structuré) |
| 2024-09 (EMNLP) | Guo et al., *Active-Dormant Attention Heads* | Cycle de renforcement mutuel : attention sur le puits ↔ norme de valeur réduite | Tâche jouet (Bigram-Backcopy), petits transformeurs | Résumé |
| 2024-07 (prépub.) | Yan, Du et al., *Unveiling and Controlling Anomalous Attention Distribution* | Sur **Llama3-8B** : le premier jeton reçoit une attention disproportionnée dès la 3ᵉ couche, et sa norme L2 dans la matrice V chute fortement dès la 2ᵉ couche | Observation + intervention causale (masque d'attention modifié) sur un jeton choisi par position | Lu en entier |
| 2024-05 (ACL) | Yao et al., SirLLM | L'entropie propre d'un jeton (surprise à sa génération) prédit l'attention qu'il reçoit ensuite | KV-cache à long terme, score d'entropie par jeton | Résumé |
| 2020 (EMNLP) | Kobayashi, Yokoi, Inui | Spearman 0,75 entre rang de fréquence et ‖f(x)‖ ; 0,06 avec l'attention | BERT, NMT | Lu en entier (recherche économe) |

## Pièges méthodologiques trouvés

- **Deux notions de « norme de valeur » coexistent** dans la littérature : la norme avant W_O (ce que
  Kobayashi appelle v) et la norme après W_O (notre ‖W_O v‖, aussi utilisée par Yan et al.). Un article à
  l'autre, ce n'est pas toujours la même quantité qui est comparée.
- **L'« amplification » et l'« atténuation » dépendent fortement du type d'injection.** Kim et al. (2026)
  injectent un **préfixe continu** persistant tout au long du contexte ; nous injectons un **jeton discret
  unique** en position fixe, puis laissons le modèle continuer librement. Leur théorème (la masse
  d'attention sur l'injection décroît avec la taille du cache) décrit la dilution d'un signal continu, pas
  l'évolution de ce que le modèle écrit après un seul jeton choquant. Les deux résultats ne se contredisent
  donc pas nécessairement, mais la comparaison directe serait un abus de généralisation à éviter.
- **Les neurones d'entropie de Stolfo et al. offrent une explication concurrente à la nôtre.** Si l'amplification
  observée dans nos 10 configurations passe par ces neurones (effet médié par LayerNorm) plutôt que par
  la norme des valeurs dans l'attention, notre explication « la valeur crie plus fort » serait incomplète.
  Ceci n'a pas été testé.

## Expérience révisée

A1k mesure déjà ρ_V (excès de valeur) et le lien avec I (amplification). Une extension testable, non
construite : ablation des neurones d'entropie (méthode de Stolfo et al., repérables par norme de poids
élevée et faible `LogitVar`) pendant la passe rejouée du choc fort, pour voir si l'amplification I disparaît.
**Critère de réfutation** pour l'explication par la norme des valeurs : si l'ablation des neurones d'entropie
seule (sans toucher à l'attention) supprime l'essentiel de l'amplification I, l'explication par le mécanisme
attentionnel (notre hypothèse) serait en grande partie écartée au profit du mécanisme par neurones.

## Lectures prioritaires

1. Stolfo et al. 2024, *Confidence Regulation Neurons in Language Models* — mécanisme concurrent direct.
2. Yan et al. 2024, *Unveiling and Controlling Anomalous Attention Distribution* — extension à Llama3-8B,
   méthode d'intervention causale transposable.
3. Kim et al. 2026, *From Noise to Diversity: Random Embedding Injection* — pour la nuance
   amplification/atténuation selon le type d'injection.

## Ce que la recherche n'a pas trouvé

- Aucun test causal, sur un LLM décodeur génératif moderne, d'un **jeton discret unique** injecté et comparé
  à un témoin apparié, avec mesure de la norme de valeur **et** de la dynamique d'entropie qui suit.
- Aucune étude primaire en français, chinois, allemand ou japonais sur cette question précise (seulement
  des reprises de travaux anglophones).
- Rien sur transformer-circuits.pub ni l'Alignment Forum (couche non couverte, outil indisponible).
- Aucune réplication manquée ni critique publiée de Kobayashi et al. 2020.
- IEEE Xplore et dblp.org non interrogés (hors budget).

## Statut épistémique

Kobayashi et al. 2020, Yan et al. 2024, Stolfo et al. 2024 et Kim et al. 2026 : lus en entier ou en rapport
structuré complet. VATP, Guo et al., SirLLM : résumés via WebSearch, non lus en entier. Tous les travaux
cités sont des prépublications arXiv sauf Kobayashi et al. (EMNLP, révisé par les pairs) et Stolfo et al.
(NeurIPS, révisé par les pairs).

## Liens avec tes recherches

- A1k (lancé/à lancer) mesure déjà ρ_V et son lien avec I : les résultats pourront être comparés directement
  à Yan et al. (attend-on la même baisse de norme sur un jeton forcé mi-séquence que sur le jeton de début
  de séquence qu'ils étudient ?).
- L'ablation des neurones d'entropie (Stolfo et al.) est une expérience discriminante faisable sur Kaggle,
  séparée d'A1k, non construite à ce jour.

## Sources

- Kobayashi, Yokoi, Inui (2020). *Attention is Not Only a Weight.* EMNLP. doi:10.18653/v1/2020.emnlp-main.574
- Yan, Du, Deng et al. (2024). *Unveiling and Controlling Anomalous Attention Distribution in Transformers.*
  arXiv:2407.01601
- Stolfo, Wu, Gurnee et al. (2024). *Confidence Regulation Neurons in Language Models.* NeurIPS. arXiv:2406.16254
- Guo, Mei et al. (2024). *Active-Dormant Attention Heads.* arXiv:2410.13835
- Kim, Lee, Yeom et al. (2026). *From Noise to Diversity: Random Embedding Injection in LLM Reasoning.*
  arXiv:2605.11936
- Yao, Li, Zhao (2024). *SirLLM: Streaming Infinite Retentive LLM.* ACL. arXiv:2405.12528
- VATP (2024). *Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction.*
  EMNLP Findings. arXiv:2406.12335

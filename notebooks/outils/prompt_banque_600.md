# Prompt à copier-coller (tout ce qui suit la ligne ===== , jusqu'à la fin du fichier)

Mode d'emploi pour Erick :
1. Remplace `XXX` (ligne PRÉFIXE) par un code court par modèle : GPT, GEM, CLA, MIS, LLA…
2. Colle tout le bloc dans le modèle. S'il s'arrête avant la fin, écris seulement : `CONTINUE`.
3. Copie toute la sortie dans un fichier texte `banque_XXX.tsv` (un fichier par modèle).
4. Mets les fichiers dans un dossier et lance `python3 verifier_banque.py dossier/` : il vérifie, retire les doublons
   et assemble une banque finale équilibrée de 600 débuts de phrase, avec son empreinte md5.

=====

# TÂCHE : produire une banque de 600 débuts de phrase pour une expérience scientifique

PRÉFIXE = XXX

## Contexte (à lire, ne pas commenter)

Tu aides un chercheur en interprétabilité des modèles de langage. Il mesure comment un modèle de langage de base
(non conversationnel) poursuit un texte. Il a besoin d'une banque de **débuts de phrase** : des phrases commencées puis
coupées au milieu, que le modèle testé devra continuer. Ce ne sont **pas des questions**. La banque sera vérifiée par un
programme : toute ligne qui ne respecte pas le format exact sera rejetée automatiquement. La précision du format compte
plus que l'originalité.

## Structure exigée : 100 thèmes × 6 débuts = 600 lignes

- **100 thèmes**, numérotés T001 à T100.
  - T001 à T050 : registre **general** (grand public), 10 domaines × 5 thèmes, dans cet ordre :
    nature ; histoire ; arts ; sport ; cuisine ; vie quotidienne ; géographie ; santé ; société ; technologie grand public.
    (T001–T005 = nature, T006–T010 = histoire, etc.)
  - T051 à T100 : registre **technique** (niveau universitaire ou spécialisé), 10 domaines × 5 thèmes, dans cet ordre :
    physique ; chimie ; biologie ; mathématiques ; informatique ; ingénierie ; médecine ; sciences de la Terre ;
    économie ; linguistique.
- Un thème est un sujet précis nommé en quelques mots (exemples de forme : « les phares maritimes », « la fermentation
  lactique », « les nombres premiers de Mersenne »). Les 100 thèmes doivent être tous différents.
- **Thèmes interdits** (déjà utilisés ; ne pas les reprendre, ni en français ni en anglais, ni sous une forme voisine) :
  photosynthèse, révolution industrielle, volcans, jazz, cuisine italienne, trous noirs, hockey sur glace, abeilles,
  intelligence artificielle, Saint-Laurent, Renaissance, marées, vélo en ville, vaccins, Grèce antique, forêts boréales,
  théâtre, électricité, échecs, chocolat, ponts suspendus, mémoire humaine, comètes, recyclage, chromodynamique
  quantique, catalyse enzymatique, réseaux de neurones convolutifs, cryptographie à clé publique, thermodynamique hors
  équilibre, résonance magnétique nucléaire, topologie algébrique, régulation épigénétique, supraconducteurs, compilation
  à la volée, turbulence, inférence bayésienne variationnelle, tectonique des plaques, ondes gravitationnelles,
  photolithographie, théorie des catégories, CRISPR, codes correcteurs d'erreurs, relativité générale, apprentissage par
  renforcement, Empire romain, tremblements de terre, blues, cuisine japonaise, étoiles à neutrons, basketball, fourmis,
  apprentissage automatique, Mississippi, courants océaniques, jardinage urbain, antibiotiques, Égypte ancienne, forêts
  tropicales, opéra, magnétisme, poker, café, gratte-ciel, sommeil, astéroïdes, énergie solaire.
- Pour **chaque thème : 6 débuts de phrase**, 3 en français (langue `fr`) puis 3 en anglais (langue `en`).
  Les 3 débuts anglais ne doivent pas être des traductions des 3 débuts français.

## Genres : 8 genres, attribués par une table fixe

Codes des genres :
- `EXP` explicatif : définit ou explique un mécanisme (« Le principe des écluses repose sur »)
- `NAR` narratif : petit récit avec un personnage fictif (« Ce matin-là, en ouvrant la porte du phare, Léa »)
- `HIS` historique : origine ou évolution (« Les premières écluses à sas sont apparues »)
- `PRO` procédural : marche à suivre (« Pour régler un dérailleur arrière, il faut d'abord »)
- `REP` reportage : style journalistique (« Selon un reportage diffusé hier soir, les apiculteurs »)
- `IDR` idée reçue : erreur courante ou fait contre-intuitif (« Contrairement à ce que l'on croit souvent, les chats »)
- `FOR` formel : formalisme, équation, définition rigoureuse (« Formellement, une matrice orthogonale est une matrice »)
- `ARG` argumentatif : thèse ou débat (« Les défenseurs de la semaine de quatre jours soutiennent que »)

Le numéro du thème décide des genres. Calcule `r` = reste de la division du numéro par 8
(T001 → 1, T008 → 0, T009 → 1, T016 → 0, T100 → 4), puis applique la ligne correspondante :

| r | fr (débuts 1, 2, 3) | en (débuts 4, 5, 6) |
|---|---|---|
| 1 | EXP, NAR, HIS | PRO, REP, IDR |
| 2 | NAR, HIS, PRO | REP, IDR, FOR |
| 3 | HIS, PRO, REP | IDR, FOR, ARG |
| 4 | PRO, REP, IDR | FOR, ARG, EXP |
| 5 | REP, IDR, FOR | ARG, EXP, NAR |
| 6 | IDR, FOR, ARG | EXP, NAR, HIS |
| 7 | FOR, ARG, EXP | NAR, HIS, PRO |
| 0 | ARG, EXP, NAR | HIS, PRO, REP |

## Règles pour chaque début de phrase (champ `texte`)

1. **Entre 6 et 16 mots.**
2. C'est une phrase **commencée et coupée au milieu** : la suite doit être ouverte, avec plusieurs continuations
   plausibles. Elle se termine par un mot, une virgule ou un deux-points. **Jamais** de point final, de point
   d'interrogation, de point d'exclamation ou de points de suspension.
3. Le thème y est **nommé ou clairement reconnaissable**.
4. Interdits : guillemets, parenthèses, tabulations, émojis, mise en forme, listes ; noms de personnes réelles
   vivantes ; noms de marques ; dates postérieures à 2020 ; contenu violent, sexuel, politique partisan ou conseil
   médical personnel.
5. Français correct (contractions : du, au, des, aux) ; anglais naturel.
6. **Variété** : dans toute la banque, au plus 5 débuts par langue peuvent commencer par les 3 mêmes mots. Ne recopie
   pas ces gabarits déjà utilisés : « Voici ce qu'il faut savoir sur », « Hier, un reportage consacré à »,
   « Pour bien comprendre », « L'histoire de … commence », « Dans ce cours avancé sur », « Une revue de littérature
   récente sur », « Le formalisme mathématique de », « Une erreur fréquente à propos de », et leurs équivalents anglais.
7. Aucun début ne doit répéter ou paraphraser un autre début de la banque.

## Format de sortie (strict)

- Première ligne, exactement :
  `id	theme_id	registre	domaine	theme_fr	theme_en	langue	genre	texte`
- Puis **une ligne par début de phrase**, champs séparés par **une tabulation** (pas d'espaces à la place), dans l'ordre :
  - `id` : PRÉFIXE-numéro du thème-numéro du début, par exemple `XXX-T001-1` … `XXX-T001-6`
  - `theme_id` : T001 … T100
  - `registre` : general ou technique
  - `domaine` : le domaine en français, écrit exactement comme dans la liste ci-dessus
  - `theme_fr`, `theme_en` : le nom du thème dans les deux langues, identique sur les 6 lignes du thème
  - `langue` : fr ou en
  - `genre` : le code de la table
  - `texte` : le début de phrase
- Thèmes dans l'ordre T001 à T100, les 6 lignes d'un thème à la suite (débuts 1 à 3 en fr, 4 à 6 en en).
- **Rien d'autre** : pas d'introduction, pas de commentaire, pas de bloc de code, pas de ligne vide.

Exemple de deux lignes valides (thème fictif, ne pas réutiliser) :
`XXX-T001-1	T001	general	nature	les lucioles	fireflies	fr	EXP	La lumière produite par les lucioles provient d'une réaction chimique qui`
`XXX-T001-4	T001	general	nature	les lucioles	fireflies	en	PRO	To photograph fireflies at dusk without a flash, you should first`

## Si tu ne peux pas tout écrire en une réponse

Arrête-toi après la dernière ligne complète d'un thème et écris seule sur la dernière ligne : `### SUITE` suivi du
prochain numéro de thème (par exemple `### SUITE T034`). Quand on te répond `CONTINUE`, reprends exactement à ce
thème, **sans** réécrire l'en-tête ni répéter de ligne, avec les mêmes règles.

## Avant d'écrire chaque thème, vérifie mentalement

- le numéro du thème → la bonne ligne de la table des genres ;
- 6 lignes, 9 champs chacune, séparés par des tabulations ;
- 6 à 16 mots, pas de ponctuation finale interdite, thème présent ;
- aucun thème interdit, aucune répétition.

Commence maintenant par la ligne d'en-tête.

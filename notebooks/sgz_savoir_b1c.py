# %% [markdown]
# # SGZ B1c — Savoir sans douter : le modèle sait-il qu'il s'est trompé, sans que sa sortie hésite ?
#
# **Point de départ.** B1 : après son erreur de calcul, le modèle recopie la mauvaise valeur dans 100 % des suites, sans
# aucune hésitation (entropie à la recopie 0,0001 chez Qwen3, comme après la bonne réponse). Sun, Stolfo et Sachan (2025,
# arXiv 2507.12379) : des sondes sur les états cachés détectent les erreurs d'addition à plus de 90 % (Gemma 2 2B IT, Phi-3),
# et la bonne réponse reste décodable quand le modèle se trompe. Hypothèse : **le modèle sait, mais sa sortie ne doute pas.**
# B1c mesure les deux sur les mêmes erreurs, **à la même position**.
#
# **Dispositif.** Les problèmes retenus par B1 (472 pour Qwen3, 600 pour SmolLM3 ; recopiés ici, même banque, même graine,
# contrôle par empreinte MD5 des prompts). Pour chaque problème, deux textes forcés : la bonne réponse c et l'erreur w que le
# modèle a réellement donnée, puis le début de la recopie (« c ; c… » ou « w ; w… », recopie observée dans 100 % des suites
# de B1). Position mesurée : juste avant le premier jeton où la recopie de w diffère de celle de c (le chiffre qui trahit
# l'erreur). À cette position, une seule passe donne :
# - **savoir** : les états cachés de toutes les couches → sonde logistique (erreur ou non), validation croisée par
#   problème (5 plis), couche choisie par validation interne sur les plis d'entraînement (pas de choix a posteriori) ;
# - **douter** : la loi de sortie → log-probabilité de l'**autre** valeur (après w : le chiffre de c ; après c : celui de w).
#   Si la sortie « sait », elle donne plus de poids au bon chiffre en recopiant l'erreur qu'au mauvais en recopiant la bonne
#   réponse.
#
# **Référence de surface.** Une sonde sur des traits superficiels (derniers chiffres de a, b et v, premier chiffre et
# longueur de v, cohérence du dernier chiffre) : si elle fait aussi bien que les états cachés, le « savoir » se réduit à une
# vérification triviale.
#
# ## Critères (fixés avant toute exécution ; bootstrap sur les problèmes, B = 1000 ; AUC = aire sous la courbe ROC)
#
# - **Contrôle** : sonde sur étiquettes permutées (au sein de chaque problème) : AUC dans [0,40 ; 0,60] ; sinon S1 NON
#   INTERPRÉTABLE.
# - **S1, le modèle sait-il ?** AUC de la sonde (hors échantillon). SAIT si la borne basse ≥ 0,70 et si l'écart avec la sonde
#   de surface a une borne basse > 0,05 ; NE SAIT PAS si la borne haute < 0,60 ; sinon INDÉTERMINÉ.
# - **S2, la sortie doute-t-elle ?** AUC du score « log-probabilité de l'autre valeur » (erreur contre bonne réponse). DOUTE
#   si la borne basse > 0,60 ; SANS DOUTE si l'IC95 ⊂ [0,40 ; 0,60] ; CERTITUDE ACCRUE si la borne haute < 0,40 ; sinon
#   INDÉTERMINÉ.
# - **Synthèse, thèse « savoir sans douter »** : SOUTENUE si S1 = SAIT et S2 ∈ {SANS DOUTE, CERTITUDE ACCRUE} pour les deux
#   modèles ; SAVOIR ET DOUTER si S1 = SAIT et S2 = DOUTE pour les deux ; RÉFUTÉE si S1 = NE SAIT PAS pour les deux ;
#   DÉPEND DU MODÈLE si les deux modèles suivent deux de ces schémas différents ; sinon INDÉTERMINÉE.
# - **Descriptifs** : AUC par couche ; couches choisies ; AUC de la surface ; entropie et log-probabilité du jeton recopié à
#   la position mesurée ; position de divergence ; part des problèmes écartés (recopie sans divergence).
# - Témoins positifs synthétiques (sonde, règles, synthèse).
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# S1 : SAIT 60 %, INDÉTERMINÉ 25 %, NE SAIT PAS 15 %. S2 : SANS DOUTE 55 %, DOUTE 25 %. Synthèse SOUTENUE 35 %.
#
# ## Limites connues
#
# - Une sonde qui réussit montre que l'information est lisible, pas que le modèle s'en sert.
# - Une seule tâche (multiplication), une seule position ; la recopie est forcée (fidèle à B1, où elle est observée à 100 %).
# - Les sondes de Sun et al. visaient le signe « = » avant la réponse ; ici la position suit la réponse écrite.
#
# ## Réglages Kaggle
#
# Accélérateur **GPU T4 x2**, **Internet activé** (aucun jeton Hugging Face requis). Durée estimée : ~10 min par modèle
# (sondes comprises) [estimation]. **À lire en priorité** : `[TÉMOIN POSITIF]`, `[VÉRIF]`, `[CONTRÔLE]`, `[S1]`, `[S2]`,
# `[SYNTHÈSE]`.

# %%
import gc
import hashlib
import json
import os
import re
import sys
import threading
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/sgz_b1c" if os.path.isdir("/kaggle/working") else "sgz_b1c_out")
HF_CACHE = os.environ.get("HF_CACHE", "/tmp/hf_cache")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

T_MAIN = 0.6
Q = 1.45
N_POOL, K_ECH, N_MAX, N_MIN = 2000, 8, 600, 100   # banque et sélection de B1 (la sélection n'est refaite que sans liste B1)
N_EXEMPLES, LONG_REP = 4, 8
PLIS, PLIS_INT, C_REG = 5, 3, 0.05
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
SEED = 20261029                                  # graine de B1 : même banque de problèmes
SEL = SEED + 31
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
MODELES_A_LANCER = ["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
MODELS = [(m.split("/")[-1], [m]) for m in MODELES_A_LANCER]
if os.environ.get("MODEL_IDS"):
    MODELS = [(m.split("/")[-1], [m]) for m in os.environ["MODEL_IDS"].split(";")]
if SMOKE_TEST:
    N_POOL, K_ECH, N_MAX, N_MIN = int(os.environ.get("SMOKE_N", "200")), 6, 80, 20
    N_EXEMPLES, LONG_REP = 2, 6
    BOOT_N = 50
    MODELS = [("smoke-qwen3-aleatoire", ["smoke-qwen3-random"]), ("smoke-qwen3-texte", ["smoke-qwen3-texte"])]
if not SMOKE_TEST and DEVICE != "cuda":
    raise RuntimeError("Aucun GPU détecté : active l'accélérateur « GPU T4 x2 » (Settings → Accelerator).")
torch.manual_seed(SEED)
np.random.seed(SEED)
import transformers
print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST} DEVICE={DEVICE} N_GPU={N_GPU} OUT={OUT}")
print(f"[CONFIG] sondes : {PLIS} plis externes, {PLIS_INT} internes, C = {C_REG} ; modèles={[m[0] for m in MODELS]}")
print(f"[CONFIG] torch {torch.__version__}, transformers {transformers.__version__}")

# %%
# ---------- Problèmes ----------
def probleme(rng, kind):
    if kind == "2x1":                            # harnais local (SMOKE_TEST) seulement
        a, b = int(rng.integers(10, 100)), int(rng.integers(2, 10))
        c, d = int(rng.integers(10, 100)), int(rng.integers(1, 10))
    else:
        a = int(rng.integers(102, 1000)) if kind == "3x2" else int(rng.integers(12, 100))
        b = int(rng.integers(12, 100))
        c, d = int(rng.integers(100, 1000)), int(rng.integers(10, 100))
    return a, b, c, d


def chaine(a, b, c, d):
    p = a * b
    return f"Calcul : {a} * {b} + {c} - {d}\nÉtapes : {a} * {b} = {p} ; {p} + {c} = {p + c} ; {p + c} - {d} = {p + c - d}\n"


KINDS = ["2x1"] if SMOKE_TEST else ["2x2", "3x2"]
rng_pb = np.random.default_rng(SEED + 2)
EXEMPLES = "\n".join(chaine(*probleme(rng_pb, KINDS[i % len(KINDS)])) for i in range(N_EXEMPLES))
POOL = [probleme(rng_pb, KINDS[i % len(KINDS)]) for i in range(N_POOL)]
TYPE = np.array([KINDS[i % len(KINDS)] for i in range(N_POOL)])
BONNE = np.array([a * b for a, b, _, _ in POOL])
PROMPTS = [EXEMPLES + "\n" + f"Calcul : {a} * {b} + {c} - {d}\nÉtapes : {a} * {b} = " for a, b, c, d in POOL]
print(f"[CONFIG] exemple de prompt :\n{PROMPTS[0]}…")
# Problèmes retenus par B1 (indice dans la banque, mauvaise réponse la plus fréquente du modèle), recopiés de ses résultats
_IDX_Q = [
    1, 5, 9, 17, 19, 21, 29, 39, 41, 45, 59, 61, 65, 69, 75, 83, 86, 87, 91, 94, 96, 103, 111, 115,
    125, 129, 131, 140, 153, 161, 163, 164, 172, 175, 177, 181, 183, 187, 191, 195, 202, 213, 227, 228, 243, 245, 249, 253,
    255, 257, 273, 281, 283, 287, 291, 292, 293, 297, 307, 317, 319, 325, 326, 329, 335, 336, 337, 349, 351, 358, 361, 363,
    369, 373, 375, 381, 389, 393, 397, 399, 403, 411, 415, 419, 424, 432, 437, 439, 445, 446, 447, 448, 454, 457, 461, 465,
    470, 475, 478, 484, 487, 489, 493, 496, 497, 498, 499, 507, 509, 511, 513, 515, 518, 519, 521, 525, 527, 528, 535, 540,
    544, 547, 549, 550, 553, 555, 560, 564, 565, 571, 575, 579, 581, 582, 583, 601, 605, 607, 625, 627, 630, 631, 635, 638,
    639, 641, 643, 646, 647, 649, 658, 661, 663, 666, 669, 673, 677, 683, 684, 685, 689, 698, 701, 703, 715, 730, 731, 735,
    747, 749, 759, 760, 763, 767, 769, 775, 776, 783, 785, 787, 791, 799, 803, 805, 806, 807, 809, 812, 813, 819, 821, 823,
    826, 834, 837, 841, 843, 845, 849, 853, 857, 863, 865, 881, 887, 889, 897, 905, 909, 915, 923, 927, 933, 934, 935, 937,
    939, 941, 945, 947, 953, 958, 961, 965, 970, 971, 973, 977, 981, 988, 993, 995, 997, 1003, 1008, 1011, 1013, 1017, 1023, 1026,
    1029, 1039, 1045, 1047, 1049, 1053, 1061, 1065, 1071, 1073, 1075, 1079, 1082, 1083, 1087, 1088, 1091, 1093, 1105, 1107, 1110, 1117, 1126, 1128,
    1131, 1133, 1135, 1145, 1151, 1156, 1159, 1160, 1164, 1166, 1171, 1179, 1187, 1189, 1191, 1197, 1201, 1204, 1209, 1211, 1214, 1215, 1217, 1222,
    1225, 1232, 1235, 1241, 1245, 1247, 1248, 1249, 1250, 1255, 1258, 1259, 1263, 1265, 1267, 1270, 1271, 1275, 1277, 1279, 1281, 1283, 1286, 1287,
    1291, 1292, 1299, 1303, 1309, 1311, 1315, 1317, 1329, 1330, 1333, 1335, 1340, 1345, 1353, 1356, 1359, 1363, 1371, 1373, 1377, 1379, 1381, 1383,
    1385, 1387, 1397, 1398, 1401, 1404, 1405, 1407, 1409, 1411, 1413, 1420, 1423, 1427, 1431, 1435, 1437, 1441, 1443, 1444, 1446, 1451, 1457, 1459,
    1461, 1467, 1479, 1481, 1487, 1490, 1491, 1499, 1518, 1519, 1520, 1521, 1532, 1540, 1544, 1545, 1547, 1549, 1551, 1557, 1560, 1569, 1571, 1579,
    1581, 1597, 1599, 1617, 1619, 1620, 1622, 1625, 1642, 1643, 1647, 1651, 1658, 1659, 1660, 1665, 1666, 1667, 1673, 1677, 1681, 1689, 1693, 1703,
    1707, 1708, 1710, 1714, 1717, 1719, 1723, 1727, 1731, 1733, 1747, 1752, 1753, 1754, 1759, 1763, 1768, 1783, 1803, 1809, 1813, 1821, 1825, 1829,
    1837, 1843, 1845, 1849, 1851, 1853, 1855, 1865, 1873, 1875, 1883, 1891, 1895, 1896, 1901, 1903, 1905, 1920, 1921, 1923, 1925, 1929, 1931, 1935,
    1947, 1951, 1953, 1957, 1961, 1965, 1968, 1969, 1974, 1983, 1985, 1987, 1991, 1993, 1994, 1999,
]
_W_Q = [
    15995, 7042, 18336, 29598, 11808, 16897, 64680, 24490, 36292, 37847, 37336, 73904, 4414, 38638, 34995, 14322,
    3861, 54456, 49185, 3966, 3286, 34496, 17585, 12144, 4283, 76239, 54132, 6312, 16112, 19376, 67152, 5182,
    4534, 31830, 56616, 55359, 22004, 30658, 28032, 83832, 3895, 11413, 47436, 8119, 15463, 40813, 24074, 32244,
    68951, 21999, 9258, 23013, 22995, 62384, 36576, 5501, 48432, 26135, 17238, 34448, 42200, 28035, 4203, 36730,
    35856, 7488, 64595, 29900, 5903, 2960, 15073, 57388, 36915, 74865, 49160, 35488, 58569, 67927, 13281, 20318,
    24342, 40432, 38566, 31690, 4717, 5862, 19344, 82899, 8118, 4758, 29556, 4504, 5156, 40476, 69994, 77248,
    7184, 29766, 3472, 4579, 24456, 49078, 5986, 5690, 9651, 4928, 21490, 37947, 30759, 18778, 16518, 34568,
    5758, 31728, 6968, 69842, 26528, 7938, 30558, 5936, 8031, 17634, 35746, 2773, 71057, 16856, 4163, 7012,
    36694, 24065, 38505, 23579, 29973, 4551, 25438, 72201, 54210, 44808, 54720, 19925, 8856, 28895, 46853, 7944,
    12034, 40128, 55290, 6566, 43944, 31225, 1702, 17419, 38066, 7180, 15598, 50217, 50239, 36942, 2005, 26504,
    29635, 7209, 67464, 48396, 5935, 1481, 59460, 21840, 16192, 53688, 16336, 3966, 29922, 12201, 50025, 15514,
    2966, 52230, 40044, 41376, 36355, 25024, 48288, 42881, 5424, 38620, 53235, 7124, 31594, 30080, 37763, 5728,
    2706, 2392, 39285, 21070, 51824, 39920, 46565, 77445, 76854, 17626, 17898, 73572, 10581, 42663, 48829, 26960,
    71472, 53445, 56048, 37656, 59159, 3997, 56184, 71668, 40664, 10257, 41100, 56964, 47868, 1481, 69894, 79140,
    3605, 33490, 9496, 7105, 68176, 5186, 20844, 63903, 18798, 6596, 2451, 10346, 55530, 18684, 31102, 4955,
    39092, 42415, 35034, 37698, 49355, 38204, 47019, 9988, 52165, 35041, 35680, 9444, 7405, 42690, 30170, 4088,
    27064, 4822, 32941, 13872, 3965, 51873, 3006, 4985, 41525, 71364, 9088, 32820, 53748, 1481, 40283, 4551,
    6358, 3803, 29968, 30576, 3106, 76971, 39769, 22272, 51664, 5571, 85904, 74968, 4247, 53145, 30816, 6663,
    15419, 7012, 32030, 52035, 5396, 52844, 6404, 13926, 4985, 27552, 4854, 14220, 26608, 19155, 38429, 4151,
    47262, 26357, 24906, 32032, 35477, 39938, 4163, 22296, 34156, 5492, 37780, 18584, 84504, 26814, 4815, 59923,
    16064, 2906, 38575, 6784, 1082, 49449, 49920, 6415, 9673, 56125, 8204, 18266, 20296, 48896, 36528, 16434,
    22599, 57865, 75640, 2946, 36644, 4791, 53330, 18215, 10528, 15311, 24982, 6713, 18553, 23942, 13185, 49990,
    31239, 49270, 12762, 6088, 6663, 55398, 54791, 41390, 21411, 39995, 52640, 58488, 38534, 5344, 18354, 19152,
    3144, 20712, 6546, 3481, 5646, 5566, 7968, 39684, 84756, 63144, 8062, 40448, 6458, 24505, 13232, 6864,
    22417, 10304, 33356, 45710, 33480, 4168, 3246, 78625, 3296, 44576, 8656, 62790, 6128, 44368, 6952, 40392,
    5068, 35032, 20700, 83832, 37615, 43936, 14385, 23008, 21052, 1618, 7045, 7288, 22450, 33736, 25456, 8764,
    49850, 16227, 20328, 1985, 47896, 7184, 29940, 45890, 5788, 29568, 78384, 83370, 11683, 16222, 21187, 40996,
    5448, 10848, 57874, 55648, 19145, 46176, 27544, 15931, 41683, 39872, 52720, 50832, 73497, 8605, 37344, 26903,
    5052, 4088, 66216, 53164, 11292, 8624, 53217, 6590, 23984, 21536, 49956, 10568, 5574, 16575, 3846, 15587,
    3538, 23824, 14071, 29764, 23622, 71333, 4088, 30480,
]
_IDX_S = [
    0, 8, 10, 11, 13, 15, 17, 19, 20, 23, 25, 26, 28, 29, 33, 35, 38, 39, 41, 42, 53, 58, 60, 61,
    64, 65, 68, 69, 85, 86, 87, 91, 92, 93, 94, 95, 96, 99, 100, 101, 103, 105, 106, 109, 111, 115, 117, 121,
    122, 123, 124, 125, 128, 133, 136, 141, 145, 146, 147, 149, 153, 157, 159, 160, 162, 164, 166, 171, 175, 177, 183, 187,
    189, 190, 191, 197, 200, 202, 203, 206, 211, 212, 213, 215, 222, 223, 224, 225, 226, 227, 228, 229, 233, 236, 237, 241,
    245, 247, 255, 257, 259, 260, 268, 269, 271, 272, 275, 279, 281, 282, 289, 291, 293, 295, 298, 306, 309, 315, 317, 318,
    319, 325, 326, 327, 329, 330, 331, 333, 341, 346, 347, 348, 349, 351, 357, 359, 365, 368, 379, 385, 386, 391, 397, 399,
    403, 406, 407, 410, 411, 415, 417, 419, 423, 425, 426, 427, 432, 434, 435, 437, 439, 441, 445, 447, 448, 451, 454, 455,
    460, 461, 464, 465, 471, 475, 477, 478, 483, 484, 486, 488, 498, 499, 507, 509, 511, 513, 517, 518, 520, 521, 527, 528,
    533, 535, 540, 541, 543, 544, 547, 549, 554, 557, 559, 562, 564, 565, 568, 569, 570, 576, 579, 581, 582, 583, 585, 587,
    589, 591, 593, 595, 597, 599, 603, 605, 607, 609, 614, 615, 616, 619, 627, 630, 637, 638, 646, 647, 649, 651, 654, 657,
    663, 664, 665, 667, 669, 672, 683, 685, 687, 689, 691, 695, 697, 701, 702, 703, 705, 709, 711, 714, 715, 724, 727, 729,
    731, 734, 736, 737, 744, 749, 753, 757, 760, 765, 767, 769, 773, 775, 776, 780, 781, 782, 784, 785, 789, 792, 793, 798,
    799, 803, 806, 807, 809, 811, 812, 813, 815, 816, 817, 819, 821, 823, 824, 826, 833, 834, 837, 840, 843, 845, 846, 851,
    854, 857, 859, 863, 864, 865, 867, 875, 877, 880, 881, 882, 883, 887, 893, 895, 900, 903, 907, 910, 917, 920, 921, 923,
    925, 926, 927, 928, 929, 932, 934, 935, 938, 939, 942, 954, 955, 957, 960, 961, 964, 968, 969, 970, 973, 974, 976, 977,
    978, 987, 988, 989, 996, 997, 998, 1000, 1002, 1003, 1005, 1007, 1008, 1009, 1015, 1017, 1019, 1021, 1023, 1024, 1025, 1027, 1028, 1029,
    1030, 1031, 1033, 1035, 1039, 1043, 1045, 1047, 1048, 1049, 1051, 1053, 1055, 1056, 1061, 1063, 1064, 1071, 1072, 1073, 1074, 1075, 1077, 1078,
    1079, 1083, 1085, 1090, 1092, 1096, 1099, 1100, 1101, 1102, 1105, 1107, 1116, 1119, 1121, 1125, 1127, 1128, 1130, 1131, 1133, 1134, 1135, 1136,
    1138, 1141, 1144, 1145, 1148, 1149, 1157, 1160, 1161, 1163, 1164, 1170, 1171, 1175, 1176, 1177, 1178, 1179, 1182, 1183, 1187, 1195, 1197, 1203,
    1204, 1205, 1207, 1209, 1211, 1213, 1214, 1219, 1221, 1223, 1226, 1228, 1229, 1232, 1233, 1237, 1238, 1239, 1241, 1245, 1249, 1250, 1255, 1256,
    1257, 1266, 1269, 1270, 1271, 1273, 1278, 1279, 1284, 1287, 1289, 1292, 1297, 1298, 1299, 1301, 1303, 1306, 1309, 1313, 1315, 1316, 1332, 1333,
    1335, 1337, 1338, 1340, 1346, 1348, 1351, 1359, 1362, 1363, 1366, 1370, 1371, 1374, 1375, 1377, 1379, 1381, 1385, 1387, 1391, 1394, 1396, 1399,
    1403, 1405, 1406, 1407, 1409, 1410, 1413, 1414, 1420, 1423, 1425, 1427, 1428, 1431, 1433, 1437, 1439, 1440, 1441, 1442, 1444, 1446, 1448, 1451,
    1455, 1459, 1461, 1463, 1464, 1465, 1471, 1472, 1473, 1477, 1478, 1481, 1483, 1487, 1489, 1491, 1495, 1500, 1502, 1505, 1507, 1508, 1509, 1512,
    1513, 1519, 1520, 1521, 1523, 1530, 1531, 1532, 1536, 1539, 1540, 1541, 1545, 1550, 1557, 1559, 1560, 1564, 1565, 1567, 1568, 1569, 1571, 1575,
]
_W_S = [
    6318, 4758, 5304, 8569, 46857, 18868, 29458, 11672, 3172, 23564, 17510, 2506, 4536, 64200, 27340, 21411,
    5712, 24420, 36312, 3448, 50424, 4047, 4954, 73744, 1372, 4384, 4306, 38538, 31248, 3891, 54276, 48765,
    3666, 27405, 4016, 33368, 3256, 9471, 4436, 20709, 34496, 12488, 4444, 20440, 17435, 12024, 20579, 23618,
    1219, 9324, 3016, 4313, 5541, 74832, 3465, 5572, 82336, 4214, 20698, 50576, 16032, 85040, 3182, 4101,
    2888, 5132, 4848, 28628, 31830, 57116, 21764, 30538, 23964, 5169, 28192, 40944, 5207, 3885, 43104, 4848,
    23092, 3120, 11533, 20084, 5536, 15812, 7810, 20435, 4075, 46956, 8119, 31392, 34454, 3368, 24684, 13155,
    40553, 10239, 69281, 22019, 40770, 3468, 2475, 19410, 28468, 2656, 27760, 7435, 23033, 4084, 2849, 36456,
    48432, 49376, 6130, 3656, 30894, 28103, 34448, 4022, 41700, 27885, 4253, 42954, 37030, 6168, 53955, 41369,
    61820, 8744, 17328, 7599, 30200, 5993, 53291, 36582, 19952, 3707, 34315, 34572, 5812, 19744, 13151, 20438,
    24342, 3742, 73098, 2324, 40632, 38366, 7702, 31790, 28155, 54690, 3104, 26820, 5902, 5088, 13878, 19344,
    82899, 64011, 8168, 29316, 4464, 38640, 5228, 10377, 8429, 69434, 6381, 77208, 42688, 29546, 54048, 3368,
    10516, 4549, 3124, 5873, 4872, 21510, 37747, 30969, 18618, 16638, 6138, 5708, 1655, 6996, 26528, 7958,
    39824, 30378, 5884, 18582, 12541, 8079, 17534, 35606, 2253, 3666, 37674, 4538, 7068, 36514, 6441, 25605,
    2730, 2748, 23429, 30063, 4621, 25338, 7537, 47332, 24063, 24832, 38610, 16132, 13546, 7426, 9775, 54990,
    44528, 19165, 2730, 60640, 6022, 44883, 19925, 8716, 7792, 7994, 6606, 43664, 31075, 47832, 5312, 13680,
    37926, 3048, 59904, 13329, 15458, 3534, 36862, 26644, 23010, 29545, 49410, 12332, 18728, 67584, 6756, 48336,
    38037, 37866, 11024, 4631, 6025, 4350, 35040, 72200, 59240, 3283, 3804, 33840, 4472, 52928, 15612, 4952,
    4016, 90720, 12165, 49725, 20570, 15674, 2916, 5765, 9124, 4172, 2785, 39864, 22747, 1777, 25560, 4277,
    24864, 47968, 5484, 38380, 52665, 16790, 7104, 31694, 78416, 1576, 11585, 30200, 37853, 5818, 6248, 2676,
    18660, 2392, 39485, 2886, 51624, 39640, 4948, 7820, 4632, 76854, 35034, 17526, 8514, 17938, 52040, 90586,
    10805, 3180, 73152, 3434, 64129, 10551, 87460, 9512, 5355, 32775, 13568, 6972, 39204, 3396, 1886, 56048,
    35410, 6782, 37836, 2037, 32536, 2322, 3987, 56184, 1807, 40404, 3196, 5336, 8316, 49458, 2675, 70034,
    3301, 2901, 14672, 3715, 9412, 1672, 4770, 7105, 2748, 7636, 5126, 84762, 3592, 18838, 2173, 5653,
    1623, 6586, 24104, 16319, 2499, 71080, 21783, 18604, 14576, 25658, 31022, 5614, 60875, 36656, 7158, 39332,
    4881, 31636, 26746, 9935, 42415, 5725, 34734, 37578, 4116, 49755, 26280, 38184, 61160, 2302, 47089, 57375,
    4911, 51835, 2595, 35291, 1552, 35920, 51344, 3223, 9354, 42730, 39858, 3230, 1744, 5158, 68448, 4168,
    6014, 1904, 33171, 13932, 7725, 21216, 6242, 68241, 45160, 5025, 4239, 41325, 71644, 2871, 8952, 2755,
    6821, 50922, 3407, 32980, 3927, 6492, 8428, 4591, 18832, 66006, 6288, 3145, 30168, 48440, 4855, 13642,
    5104, 30516, 2586, 38240, 3216, 17340, 22132, 6759, 5431, 33965, 24944, 86464, 74848, 63390, 4177, 59338,
    20958, 47670, 7428, 3420, 46375, 7048, 6024, 8215, 2353, 40170, 52065, 5356, 13946, 5025, 27532, 3952,
    9104, 3126, 15367, 4181, 47166, 26224, 5988, 31952, 4670, 22016, 34860, 5422, 45744, 5653, 38220, 12554,
    18584, 3703, 85144, 5418, 4770, 2349, 6235, 38625, 6834, 28528, 3154, 1092, 2136, 3353, 7688, 9603,
    6120, 56325, 5663, 3054, 8134, 4835, 53112, 20296, 48416, 37088, 22569, 57185, 30854, 3782, 6786, 17382,
    36944, 53470, 5338, 18265, 10468, 5964, 24942, 4099, 6703, 18433, 9126, 23782, 5816, 13035, 9832, 31059,
    11555, 4355, 49370, 3966, 6072, 6623, 5466, 55198, 36435, 41510, 21561, 15024, 3486, 19096, 29885, 2987,
    4357, 26610, 7544, 58808, 49459, 38774, 10155, 18354, 13432, 3808, 3598, 29680, 25492, 2811, 17111, 3347,
    30960, 20736, 6496, 3369, 19536, 7729, 34521, 5574, 5144, 7752, 5556, 14274, 39444, 3396, 40488, 20709,
    6468, 2822, 28310, 26016, 3306, 24505, 13152, 47225,
]
ITEMS_B1 = {"Qwen3-1.7B-Base": (_IDX_Q, _W_Q), "SmolLM3-3B-Base": (_IDX_S, _W_S)}
MD5_PROMPTS_B1 = "653e5b7dff900c6395fd5ab3b4278e92"
if not SMOKE_TEST:
    _md5 = hashlib.md5("\n".join(PROMPTS).encode()).hexdigest()
    print(f"[VÉRIF] empreinte des prompts : {_md5} (B1 : {MD5_PROMPTS_B1})")
    if _md5 != MD5_PROMPTS_B1:
        raise RuntimeError("la banque de problèmes diffère de celle de B1 : les problèmes recopiés ne correspondent plus")

# %%
# ---------- Mesures, règles de décision et témoins positifs ----------
@torch.no_grad()
def entropies_from_logits(logits, qs):
    """Entropies du softmax complet (float64). q = 1 : Shannon (nats) ; sinon Tsallis S_q = (1 − Σ p^q)/(q − 1)."""
    logp = torch.log_softmax(logits.double(), dim=-1)
    p = logp.exp()
    out = {}
    for q in qs:
        if q == 1.0:
            out[q] = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
        else:
            out[q] = (1.0 - (p ** q).sum(-1)) / (q - 1.0)
    return out


def gumbel_noise(V, seeds, device):
    """Bruit de Gumbel par ligne, graine explicite par ligne (mêmes nombres pour toutes les branches d'un problème)."""
    g = torch.Generator(device=device)
    rows = []
    for s in seeds:
        g.manual_seed(int(s))
        rows.append(torch.rand(V, generator=g, device=device, dtype=torch.float32))
    u = torch.stack(rows).clamp_(1e-10, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (float("nan"), float("nan"))


def ajuster(Xtr, ytr):
    """Sonde logistique sur variables standardisées ; renvoie la fonction de score."""
    sc = StandardScaler().fit(np.asarray(Xtr, np.float32))
    m = LogisticRegression(C=C_REG, max_iter=3000).fit(sc.transform(np.asarray(Xtr, np.float32)), ytr)
    return lambda X: m.decision_function(sc.transform(np.asarray(X, np.float32)))


def oof_couche(X, y, grp, plis=PLIS):
    """Scores hors échantillon à une couche fixée (X : (n, d))."""
    s = np.zeros(len(y))
    for tr, te in GroupKFold(plis).split(X, y, grp):
        s[te] = ajuster(X[tr], y[tr])(X[te])
    return s


def oof_emboite(H, y, grp, couches):
    """H : (n, L, d). Pour chaque pli externe, couche choisie par validation interne sur les plis d'entraînement seulement."""
    s, choix = np.zeros(len(y)), []
    for tr, te in GroupKFold(PLIS).split(H[:, 0], y, grp):
        meilleure, auc_m = couches[0], -1.0
        for c in couches:
            si = np.zeros(len(tr))
            for itr, ite in GroupKFold(PLIS_INT).split(tr, y[tr], grp[tr]):
                si[ite] = ajuster(H[tr[itr], c], y[tr[itr]])(H[tr[ite], c])
            a_ = roc_auc_score(y[tr], si)
            if a_ > auc_m:
                meilleure, auc_m = c, a_
        s[te] = ajuster(H[tr, meilleure], y[tr])(H[te, meilleure])
        choix.append(int(meilleure))
    return s, choix


def auc_boot(scores, y, grp, rng, autres=None):
    """AUC et IC95 par bootstrap sur les groupes ; avec « autres », IC95 de l'écart AUC(scores) − AUC(autres)."""
    g_u = np.unique(grp)
    par = {k: np.nonzero(grp == k)[0] for k in g_u}
    a0 = roc_auc_score(y, scores)
    bs, bd = [], []
    for _ in range(BOOT_N):
        ix = np.concatenate([par[k] for k in rng.choice(g_u, len(g_u))])
        if len(np.unique(y[ix])) < 2:
            continue
        a_ = roc_auc_score(y[ix], scores[ix])
        bs.append(a_)
        if autres is not None:
            bd.append(a_ - roc_auc_score(y[ix], autres[ix]))
    out = {"auc": float(a0), "ci": ci(bs)}
    if autres is not None:
        out.update({"ecart": float(a0 - roc_auc_score(y, autres)), "ecart_ci": ci(bd)})
    return out


def regle_s1(sonde, controle_ok, interpretable=True):
    c, e = sonde["ci"], sonde["ecart_ci"]
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not controle_ok:
        return "NON INTERPRÉTABLE (contrôle des étiquettes permutées raté)"
    if c[0] >= 0.70 and e[0] > 0.05:
        return "SAIT"
    if c[1] < 0.60:
        return "NE SAIT PAS"
    return "INDÉTERMINÉ"


def regle_s2(sortie, interpretable=True):
    c = sortie["ci"]
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if c[0] > 0.60:
        return "DOUTE"
    if c[0] >= 0.40 and c[1] <= 0.60:
        return "SANS DOUTE"
    if c[1] < 0.40:
        return "CERTITUDE ACCRUE"
    return "INDÉTERMINÉ"


def synthese(s1, s2, interpretable=True):
    """s1, s2 : dict modèle → verdict."""
    ms = list(s1)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    schema = {}
    for m in ms:
        if s1[m] == "SAIT" and s2.get(m) in ("SANS DOUTE", "CERTITUDE ACCRUE"):
            schema[m] = "SOUTENUE"
        elif s1[m] == "SAIT" and s2.get(m) == "DOUTE":
            schema[m] = "SAVOIR ET DOUTER"
        elif s1[m] == "NE SAIT PAS":
            schema[m] = "RÉFUTÉE"
        else:
            schema[m] = None
    if any(v is None for v in schema.values()):
        return "INDÉTERMINÉE"
    vals = set(schema.values())
    return vals.pop() if len(vals) == 1 else "DÉPEND DU MODÈLE"


def surface(a, b, v):
    """Traits superficiels : derniers chiffres de a, b, v ; premier chiffre et longueur de v ; cohérence du dernier chiffre."""
    oh = lambda k, n: np.eye(n)[k]
    return np.concatenate([oh(v % 10, 10), oh(a % 10, 10), oh(b % 10, 10), oh(int(str(v)[0]), 10), oh(min(len(str(v)), 7), 8),
                           [float(((a % 10) * (b % 10)) % 10 == v % 10)]])


# Témoins positifs : sonde emboîtée sur des états synthétiques (signal dans une seule couche), règles et synthèse
rng_pc = np.random.default_rng(SEED + 1)
n_pc, d_pc = 600, 48
grp_pc = np.repeat(np.arange(n_pc // 2), 2)
y_pc = np.tile([0, 1], n_pc // 2)
H_pc = rng_pc.normal(0, 1, (n_pc, 4, d_pc)).astype(np.float32)
u_pc = rng_pc.normal(0, 1, d_pc)
H_pc[:, 2] += np.outer(1.5 * (y_pc - 0.5), u_pc / np.linalg.norm(u_pc))      # écart de 1,5 écart-type entre classes
s_pc, choix_pc = oof_emboite(H_pc, y_pc, grp_pc, [0, 1, 2, 3])
s0_pc = oof_couche(rng_pc.normal(0, 1, (n_pc, 10)), y_pc, grp_pc)
fort = auc_boot(s_pc, y_pc, grp_pc, rng_pc, s0_pc)
nul_s, _ = oof_emboite(rng_pc.normal(0, 1, (n_pc, 2, d_pc)).astype(np.float32), y_pc, grp_pc, [0, 1])
nul = auc_boot(nul_s, y_pc, grp_pc, rng_pc, s0_pc)
sortie_nulle = auc_boot(rng_pc.normal(0, 1, n_pc), y_pc, grp_pc, rng_pc)
sortie_forte = auc_boot(rng_pc.normal(0, 1, n_pc) + 1.0 * y_pc, y_pc, grp_pc, rng_pc)
sortie_inverse = auc_boot(rng_pc.normal(0, 1, n_pc) - 1.0 * y_pc, y_pc, grp_pc, rng_pc)
checks = [("sonde : couche du signal choisie", choix_pc.count(2) >= 4, True),
          ("S1 sait", regle_s1(fort, True), "SAIT"), ("S1 ne sait pas", regle_s1(nul, True), "NE SAIT PAS"),
          ("S1 contrôle raté", regle_s1(fort, False), "NON INTERPRÉTABLE (contrôle des étiquettes permutées raté)"),
          ("S2 sans doute", regle_s2(sortie_nulle), "SANS DOUTE"), ("S2 doute", regle_s2(sortie_forte), "DOUTE"),
          ("S2 certitude accrue", regle_s2(sortie_inverse), "CERTITUDE ACCRUE"),
          ("synthèse soutenue", synthese({"a": "SAIT", "b": "SAIT"}, {"a": "SANS DOUTE", "b": "CERTITUDE ACCRUE"}), "SOUTENUE"),
          ("synthèse savoir et douter", synthese({"a": "SAIT", "b": "SAIT"}, {"a": "DOUTE", "b": "DOUTE"}), "SAVOIR ET DOUTER"),
          ("synthèse réfutée", synthese({"a": "NE SAIT PAS", "b": "NE SAIT PAS"}, {"a": "SANS DOUTE", "b": "DOUTE"}), "RÉFUTÉE"),
          ("synthèse dépend", synthese({"a": "SAIT", "b": "SAIT"}, {"a": "SANS DOUTE", "b": "DOUTE"}), "DÉPEND DU MODÈLE"),
          ("synthèse indéterminée", synthese({"a": "SAIT", "b": "INDÉTERMINÉ"}, {"a": "SANS DOUTE", "b": "SANS DOUTE"}),
           "INDÉTERMINÉE")]
print(f"[TÉMOIN POSITIF] sonde synthétique : AUC {fort['auc']:.3f} IC95 [{fort['ci'][0]:.3f} ; {fort['ci'][1]:.3f}], couches "
      f"choisies {choix_pc} (signal en couche 2) ; sans signal : AUC {nul['auc']:.3f}")
for nom, got, att in checks:
    print(f"[TÉMOIN POSITIF] {nom:<34} : {str(got):<58} (attendu {att})")
POSITIVE_CONTROL_OK = all(g == e for _, g, e in checks)
print(f"[TÉMOIN POSITIF] {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON INTERPRÉTABLE'}")

# %%
# ---------- Chargement ----------
class SmokeTokenizer:
    """Tokenizer jouet par caractères, aller-retour exact (SMOKE_TEST)."""
    ALPHA = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-«»\n+*="
             "àâäçéèêëîïôöùûüÀÇÉÈÊ")

    def __init__(self, V):
        self.V, self.pad_token_id, self.eos_token_id = V, 0, 1
        self.all_special_ids = [0, 1]

    def __len__(self):                          # jetons valides : l'alphabet (le reste du vocabulaire est masqué)
        return 2 + len(self.ALPHA)

    def encode(self, s, **kw):
        return [2 + self.ALPHA.index(c) if c in self.ALPHA else 2 + self.ALPHA.index("?") for c in s]

    def decode(self, ids, **kw):
        return "".join(self.ALPHA[i - 2] if 2 <= i < 2 + len(self.ALPHA) else "?" for i in ids)


def transformers_major():
    return int(transformers.__version__.split(".")[0])


def load_model(model_id, gpu=0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == "smoke-qwen3-random":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
        torch.manual_seed(SEED)
        return Qwen3ForCausalLM(cfg).eval().to(DEVICE), SmokeTokenizer(512), "fp32 aléatoire (smoke)"
    if model_id == "smoke-qwen3-texte":
        # Qwen3 minuscule entraîné par caractères sur des chaînes de calcul : de vraies erreurs et de vraies bonnes réponses
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512)
        torch.manual_seed(SEED)
        m, tk = Qwen3ForCausalLM(cfg).to(DEVICE), SmokeTokenizer(512)
        rng = np.random.default_rng(SEED + 99)
        corpus = np.array(tk.encode("\n".join(chaine(*probleme(rng, "2x1")) for _ in range(4000))))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for _ in range(int(os.environ.get("SMOKE_PAS", "1500"))):          # fenêtres de 192 caractères : tout le prompt
            d = rng.integers(0, len(corpus) - 193, 32)
            x = torch.tensor(np.stack([corpus[i:i + 192] for i in d]), device=DEVICE)
            loss = m(input_ids=x, labels=x).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m.eval(), tk, f"fp32 entraîné sur des chaînes de calcul (smoke, perte finale {loss.item():.2f})"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    dkey = "dtype" if transformers_major() >= 5 else "torch_dtype"
    kw = {"cache_dir": HF_CACHE}
    if DEVICE == "cpu":
        kw[dkey], mode = torch.float32, "fp32 CPU"
    else:
        kw[dkey], kw["device_map"], mode = torch.float16, {"": gpu}, f"fp16 GPU {gpu}"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if DEVICE == "cuda":
        assert next(model.parameters()).dtype == torch.float16, "type de poids inattendu"
    return model, tok, mode

# %%
# ---------- Génération (comme A1c–A1z) ----------
class Gen:
    def __init__(self, model, tok, label):
        self.model, self.tok, self.label = model, tok, label
        self.dev = model.get_input_embeddings().weight.device
        self.V = model.get_output_embeddings().weight.shape[0]
        forbid = set(tok.all_special_ids or []) | {tok.eos_token_id}
        ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        forbid |= set(ge if isinstance(ge, (list, tuple)) else ([ge] if ge is not None else []))
        forbid.discard(None)
        self.forbid, self.n_tok = sorted(forbid), len(tok)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        npar = sum(p.numel() for p in model.parameters())
        self.budget = 48 if npar < 3e9 else 32
        self._dec = {}

    def dec(self, t):
        t = int(t)
        if t not in self._dec:
            self._dec[t] = self.tok.decode([t])
        return self._dec[t]

    def left_pad(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.dev), attn.to(self.dev), L

    @torch.no_grad()
    def sample(self, ctxs, prompt_of, T, nsteps, step0, base=SEED):
        """Graine du pas s de la ligne j : base + 100003·prompt_of[j] + step0 + s (nombres aléatoires communs)."""
        n = len(ctxs)
        S, tk = np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int64)
        ids, attn, _ = self.left_pad(ctxs)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        past = None
        for s in range(nsteps):
            out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"logits non finis à T={T}, pas {s}")
            e1 = entropies_from_logits(z, [Q])[Q].cpu().numpy()
            seeds = [base + 100003 * int(prompt_of[j]) + step0 + s for j in range(n)]
            score = z / T + gumbel_noise(self.V, seeds, z.device)
            score[:, self.forbid] = -float("inf")
            if self.V > self.n_tok:
                score[:, self.n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            S[:, s], tk[:, s] = e1, nxt.cpu().numpy()
            ids = nxt[:, None].to(self.dev)
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.dev)], dim=1)
            pos = pos[:, -1:] + 1
        del past, out
        return S, tk

# %%
# ---------- Sélection de secours (comme B1) ----------
@torch.no_grad()
def tirer_reponses(g, enc, K):
    """K réponses libres à l'étape 1 par problème (bruit SEL, graine propre à chaque tirage). Valeur lue = les chiffres suivis
    de « ; » (None si illisible)."""
    P = len(enc)
    vals = [[None] * K for _ in range(P)]
    jobs = [(p, j) for p in range(P) for j in range(K)]
    nb = (len(jobs) + g.budget - 1) // g.budget
    for b, b0 in enumerate(range(0, len(jobs), g.budget)):
        t0 = time.time()
        lot = jobs[b0:b0 + g.budget]
        _, tk = g.sample([enc[p] for p, _ in lot], [p * 64 + j for p, j in lot], T_MAIN, LONG_REP, 0, base=SEL)
        for (p, j), row in zip(lot, tk):
            m = re.match(r"(\d+) ;", g.tok.decode(row.tolist()))
            vals[p][j] = int(m.group(1)) if m else None
        if b == 0:
            est = (time.time() - t0) * nb / 60
            print(f"[PROGRESSION] {g.label} sélection : premier lot en {time.time() - t0:.1f} s ; estimation ~{est:.0f} min", flush=True)
            if est / 60 > MAX_HOURS:
                raise TimeoutError(f"estimation {est:.0f} min > budget {MAX_HOURS} h")
    return vals


# %%
# ---------- Position de divergence de la recopie : états cachés et loi de sortie ----------
def recopie(tok, prompt, enc_p, v):
    """Jetons de « v ; » (réponse forcée) et de la recopie « v », découpés comme dans le texte complet."""
    t1, t2 = tok.encode(prompt + f"{v} ;"), tok.encode(prompt + f"{v} ; {v}")
    if t1[:len(enc_p)] != enc_p or t2[:len(t1)] != t1:
        return None
    return t1[len(enc_p):], t2[len(t1):]


@torch.no_grad()
def passe(g, seqs, autre, propre):
    """Pour chaque séquence (préfixe jusqu'à la position mesurée) : états cachés de toutes les couches à la dernière position,
    log-probabilités du jeton « autre » et du jeton « propre », entropie de Tsallis de la loi, jeton le plus probable."""
    head = g.model.get_output_embeddings()
    n = len(seqs)
    H, lp_a, lp_p, S, top = [], np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    for b0 in range(0, n, g.budget):
        loc = list(range(b0, min(n, b0 + g.budget)))
        ids, attn, _ = g.left_pad([seqs[i] for i in loc])
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        out = g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False, output_hidden_states=True)
        H.append(torch.stack([h[:, -1, :] for h in out.hidden_states], 1).float().cpu().numpy().astype(np.float16))
        z = head(out.last_hidden_state[:, -1, :].to(head.weight.device)).float()
        lp = torch.log_softmax(z.double(), dim=-1)
        e = entropies_from_logits(z, [Q])[Q].cpu().numpy()
        for j, i in enumerate(loc):
            lp_a[i], lp_p[i] = float(lp[j, autre[i]]), float(lp[j, propre[i]])
            S[i], top[i] = float(e[j]), int(torch.argmax(lp[j]))
        del out
    return np.concatenate(H), lp_a, lp_p, S, top

# %%
# ---------- Boucle principale ----------
interpretable = not SMOKE_TEST
ok = interpretable and POSITIVE_CONTROL_OK
ALL = {}


class TropPeu(Exception):
    pass


VERROU_RESUME, VERROU_CHARGEMENT = threading.Lock(), threading.Lock()


def ecrire_resume():
    with VERROU_RESUME, open(os.path.join(OUT, "resume_b1c.json"), "w") as fh:
        json.dump({"smoke": SMOKE_TEST, "positive_control_ok": POSITIVE_CONTROL_OK, **ALL}, fh,
                  ensure_ascii=False, indent=1, default=str)


def fmt_auc(r):
    return f"{r['auc']:.3f} IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}]"


def traiter(label, ids_m, gpu=0):
    """Un modèle, de bout en bout, sur le GPU « gpu » (corps de l'ancienne boucle sur MODELS)."""
    if DEVICE == "cuda":
        torch.cuda.set_device(gpu)
    t_start = time.time()
    model = tok = g = None
    try:
        for mid in ids_m:
            print(f"\n[CHARGEMENT] {label} : téléchargement / chargement de {mid}…", flush=True)
            try:
                with VERROU_CHARGEMENT:                # un chargement à la fois (état global de transformers)
                    model, tok, mode = load_model(mid, gpu)
                print(f"[CHARGEMENT] {label} : {mid} en {mode}")
                break
            except Exception as e:
                print(f"[CHARGEMENT] échec {mid} : {type(e).__name__}: {str(e)[:300]}")
                model = None
        if model is None:
            raise RuntimeError("modèle non chargé")
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        g = Gen(model, tok, label)
        enc = [tok.encode(p) for p in PROMPTS]
        a = {}
        if label in ITEMS_B1 and not SMOKE_TEST:
            idx0, w0 = ITEMS_B1[label]
            origine = "problèmes retenus par B1"
        else:                                    # pas de liste B1 pour ce modèle : même sélection que B1
            vals = tirer_reponses(g, enc, K_ECH)
            idx0, w0 = [], []
            for i, r in enumerate(vals):
                faux = [v for v in r if v is not None and v != BONNE[i]]
                if BONNE[i] in r and faux and len(idx0) < N_MAX:
                    idx0.append(i)
                    w0.append(max(set(faux), key=lambda v: (faux.count(v), -faux.index(v))))
            origine = "sélection refaite comme dans B1"
        seqs_c, seqs_w, own_c, own_w, keep, jdiv, wv = [], [], [], [], [], [], []
        ko_tok = ko_div = 0
        for i, w in zip(idx0, w0):
            rc, rw = recopie(tok, PROMPTS[i], enc[i], int(BONNE[i])), recopie(tok, PROMPTS[i], enc[i], int(w))
            if rc is None or rw is None:
                ko_tok += 1
                continue
            (fc, cc), (fw, cw) = rc, rw
            j = next((k for k in range(min(len(cc), len(cw))) if cc[k] != cw[k]), None)
            if j is None:
                ko_div += 1
                continue
            seqs_c.append(enc[i] + fc + cc[:j])
            seqs_w.append(enc[i] + fw + cw[:j])
            own_c.append(cc[j])
            own_w.append(cw[j])
            keep.append(i)
            jdiv.append(j)
            wv.append(int(w))
        n = len(keep)
        print(f"[SÉLECTION] {label} : {len(idx0)} problèmes ({origine}) ; retenus {n} ; découpage incohérent {ko_tok} ; recopie "
              f"sans divergence {ko_div} ; position de divergence dans la recopie : médiane {np.median(jdiv) if jdiv else float('nan'):.0f}")
        a["selection"] = {"origine": origine, "problemes": len(idx0), "retenus": n, "decoupage_incoherent": ko_tok,
                          "sans_divergence": ko_div}
        if n < N_MIN:
            a["S1"] = {"verdict": "NON TESTABLE (trop peu de problèmes)"}
            a["S2"] = {"verdict": "NON TESTABLE (trop peu de problèmes)"}
            ALL[label] = a
            print(f"[S1] {label} : {a['S1']['verdict']}")
            raise TropPeu
        if ko_tok > 0.1 * max(1, len(idx0)):
            raise RuntimeError("découpage en jetons incohérent pour plus de 10 % des problèmes : positions non fiables")
        t2 = time.time()
        H, lp_a, lp_p, S, top = passe(g, seqs_c + seqs_w, own_w + own_c, own_c + own_w)
        y = np.r_[np.zeros(n, int), np.ones(n, int)]
        grp = np.r_[np.arange(n), np.arange(n)]
        own = np.r_[own_c, own_w]
        fid = float(np.mean(top == own))
        print(f"[VÉRIF] {label} : passe aux positions de divergence {time.time() - t2:.0f} s ; le jeton le plus probable est le "
              f"jeton de la recopie dans {fid:.1%} des cas (B1 : recopie dans 100 % des suites)", flush=True)
        a["fidelite_recopie"] = fid

        # S2 : la sortie doute-t-elle ? (log-probabilité de l'autre valeur)
        rngB = np.random.default_rng(SEED + 3)
        s2r = auc_boot(lp_a, y, grp, rngB)
        s2 = regle_s2(s2r, ok)
        ent = auc_boot(S, y, grp, rngB)

        # S1 : le modèle sait-il ? (sonde emboîtée)
        t3 = time.time()
        nL = H.shape[1] - 1
        couches = list(range(1, nL + 1))
        oof, choix = oof_emboite(H, y, grp, couches)
        F = np.stack([surface(int(POOL[i][0]), int(POOL[i][1]), int(BONNE[i])) for i in keep]
                     + [surface(int(POOL[i][0]), int(POOL[i][1]), w) for i, w in zip(keep, wv)])
        oof_s = oof_couche(F, y, grp)
        sonde = auc_boot(oof, y, grp, rngB, oof_s)
        surf = auc_boot(oof_s, y, grp, rngB)
        c_mod = max(set(choix), key=choix.count)
        rng_p = np.random.default_rng(SEED + 9)
        sw = rng_p.random(n) < 0.5
        y_p = y.copy()
        y_p[:n][sw], y_p[n:][sw] = 1, 0
        perm = roc_auc_score(y_p, oof_couche(H[:, c_mod], y_p, grp))
        controle_ok = 0.40 <= perm <= 0.60
        par_couche = [float(roc_auc_score(y, oof_couche(H[:, c], y, grp))) for c in couches]
        s1 = regle_s1(sonde, controle_ok, ok)
        print(f"[MESURE] {label} sondes : {time.time() - t3:.0f} s", flush=True)
        print(f"\n{'=' * 100}\n[RÉSULTATS] {label} : {n} problèmes (erreur et bonne réponse)")
        print(f"[CONTRÔLE] {label} sonde sur étiquettes permutées (couche {c_mod}) : AUC {perm:.3f} → {'OK' if controle_ok else 'RATÉ'}")
        print(f"[S1] {label} sonde sur les états cachés (couches choisies par validation interne : {choix}) : AUC {fmt_auc(sonde)} ; "
              f"surface {fmt_auc(surf)} ; écart {sonde['ecart']:+.3f} IC95 [{sonde['ecart_ci'][0]:+.3f} ; {sonde['ecart_ci'][1]:+.3f}] "
              f"→ {s1}")
        print(f"[COUCHES] {label} AUC par couche (1 … {nL}) : " + " ".join(f"{x:.2f}" for x in par_couche))
        print(f"[S2] {label} sortie : log-probabilité de l'autre valeur, AUC {fmt_auc(s2r)} → {s2} ; médianes : après l'erreur "
              f"{np.median(lp_a[n:]):.2f}, après la bonne réponse {np.median(lp_a[:n]):.2f} nats")
        print(f"[DESCRIPTIF] {label} entropie de la loi à la position mesurée : erreur {np.mean(S[n:]):.5f}, bonne réponse "
              f"{np.mean(S[:n]):.5f} (AUC {fmt_auc(ent)}) ; log-probabilité du jeton recopié : erreur {np.median(lp_p[n:]):.4f}, "
              f"bonne réponse {np.median(lp_p[:n]):.4f}")
        a.update({"n": n, "S1": {"verdict": s1, "sonde": sonde, "surface": surf, "couches_choisies": choix, "controle_permute": perm,
                                 "auc_par_couche": par_couche},
                  "S2": {"verdict": s2, "sortie": s2r, "entropie": ent,
                         "lp_autre_med": {"erreur": float(np.median(lp_a[n:])), "bonne": float(np.median(lp_a[:n]))}},
                  "minutes": (time.time() - t_start) / 60})
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_b1c.npz"), idx=np.array(keep), w=np.array(wv), jdiv=np.array(jdiv),
                            y=y, oof=oof, oof_surface=oof_s, lp_autre=lp_a, lp_propre=lp_p, S=S,
                            H_couche=H[:, c_mod].astype(np.float16))
        print(f"[DURÉE] {label} : {a['minutes']:.1f} min", flush=True)
    except TropPeu:
        pass
    except Exception as e:
        print(f"[ERREUR] {label} : {type(e).__name__}: {str(e)[:500]}", flush=True)
        ALL[label] = {"erreur": f"{type(e).__name__}: {str(e)[:300]}"}
    del model, tok, g
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ecrire_resume()


def file_gpu(gpu, travaux):
    for label, ids_m in travaux:
        traiter(label, ids_m, gpu)


# Un modèle par GPU, en même temps (Kaggle T4 × 2) : environ deux fois moins de quota que l'un après l'autre.
# Mêmes calculs, mêmes graines : seul l'ordre des lignes du journal change (chaque ligne porte le nom du modèle).
# PARALLELE=0 pour l'ancien ordre séquentiel.
N_PAR = min(N_GPU, len(MODELS)) if DEVICE == "cuda" and os.environ.get("PARALLELE", "1") == "1" else 1
if N_PAR > 1:
    print(f"[CONFIG] {len(MODELS)} modèles en parallèle sur {N_PAR} GPU : "
          + " ; ".join(f"{l} → GPU {i % N_PAR}" for i, (l, _) in enumerate(MODELS)), flush=True)
    fils = [threading.Thread(target=file_gpu, args=(j, MODELS[j::N_PAR])) for j in range(N_PAR)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    ordre = {l: ALL.pop(l) for l, _ in MODELS if l in ALL}       # ordre du résumé : celui de MODELS
    ALL.update(ordre)
    ecrire_resume()
else:
    for label, ids_m in MODELS:
        traiter(label, ids_m)

# %%
# ---------- Synthèse ----------
print(f"\n{'=' * 100}\n[SYNTHÈSE] B1c — savoir sans douter : le modèle sait-il qu'il s'est trompé, sans que sa sortie hésite ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "S1" in ALL[m]]
v = {k: {m: ALL[m][k]["verdict"] for m in labs} for k in ("S1", "S2")}
for m in labs:
    print(f"[SYNTHÈSE] {m:<22} : S1 {v['S1'][m]:<44} ; S2 {v['S2'][m]}")
these = "NON TESTABLE (modèle manquant)" if len(labs) < len(MODELS) else synthese(v["S1"], v["S2"], ok)
ALL["these"] = these
print(f"[SYNTHÈSE] thèse « savoir sans douter » : {these} (prédiction SOUTENUE 35 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")

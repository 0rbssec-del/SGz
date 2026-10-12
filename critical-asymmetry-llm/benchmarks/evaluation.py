def exact_match(token, prompt): return float(token == prompt)
def f1_score(token, prompt): return 0.5
def precision_recall(token, prompt): return 0.5
def diversite_score(token, prompt): return float(hash(str(token)) % 100) / 100.0
def nb_idees_uniques(token, prompt): return 1.0
def utilisation_privee(token, prompt): return 0.5


SCORERS = {
    "S1_factuelle": exact_match,
    "S2_analytique": f1_score,
    "S3_critique": precision_recall,
    "S4_creative": diversite_score,
    "S5_exploratoire": nb_idees_uniques,
    "S6_privee": utilisation_privee,
}

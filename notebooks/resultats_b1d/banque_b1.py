SEED = 20261029
N_POOL = 2000
N_EXEMPLES, LONG_REP = 4, 8
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

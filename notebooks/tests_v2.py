"""Tests des volets v2 sur données synthétiques à réponse connue."""
import os, sys
os.environ["SMOKE_TEST"] = "0"; os.environ["OUT"] = "/tmp/sgz_v2_tests"
sys.path.insert(0, os.path.dirname(__file__))
from build_notebook import cells_from_script
import numpy as np
code = [c for c in cells_from_script(os.path.join(os.path.dirname(__file__), "sgz_criticite_v2.py")) if c.cell_type == "code"]
ns = {}
src = code[0].source.replace('raise RuntimeError("Aucun GPU', 'print("(test CPU)") or None and RuntimeError("Aucun GPU')
exec(src, ns)
for c in code[1:4]:
    exec(c.source, ns)
for c in code:
    if "analyse_v2" in c.source and "def analyse_v2" in c.source:
        exec(c.source.split("\n# %%")[0] if False else c.source[c.source.index("# ---------- Volets v2"):], ns)
T = np.array(ns["T_GRID"]); nT, P, L = len(T), 64, 512
rng = np.random.default_rng(0)

def make(S_fn, sur_fn):
    S = np.stack([S_fn(i, t) for i, t in enumerate(T)])
    res = {"S_p1": {q: S for q in ns["Q_LIST"]}, "surprisal": np.stack([sur_fn(i, t) for i, t in enumerate(T)])}
    return res

def ar(phi, n=(P, L)):
    x = np.zeros(n); e = rng.normal(size=n)
    for k in range(1, n[1]): x[:, k] = phi * x[:, k - 1] + e[:, k]
    return x

# Cas 1 : surprisal iid partout -> W2 RÉFUTÉ ou INDÉTERMINÉ (pas de croissance) ; corrélée à T=1.2 -> SOUTENU
res = make(lambda i, t: rng.normal(size=(P, L)) * np.exp(-(t - 1.0) ** 2 / 0.1),
           lambda i, t: rng.normal(size=(P, L)) * 3)
v = ns["analyse_v2"](res, True); print("iid :", v["W2"]["verdict"], round(v["W2"]["delta_b"], 3), v["W2"]["ci"])
assert v["W2"]["verdict"] != "SOUTENU"
res = make(lambda i, t: rng.normal(size=(P, L)),
           lambda i, t: (ar(0.995) * 0.3 if abs(t - 1.2) < 1e-9 else rng.normal(size=(P, L))) * 3)
v = ns["analyse_v2"](res, True); print("corrélé à 1,2 :", v["W2"]["verdict"], v["W2"]["T_star_chi"], round(v["W2"]["delta_b"], 3))
assert v["W2"]["verdict"] == "SOUTENU" and v["W2"]["T_star_chi"] == 1.2

# Cas 2 : W3. Dynamique pure : même dispersion à toutes les T, lissage temporel variable -> pic dynamique
def smooth_S(i, t):
    phi = 0.99 if abs(t - 1.0) > 0.25 else 0.0     # séquences lisses sauf près de 1 : G1 culmine à 1, dispersion constante
    x = ar(phi); return x / x.std(axis=1, keepdims=True)
res = make(smooth_S, lambda i, t: rng.normal(size=(P, L)))
v = ns["analyse_v2"](res, True); print("W3 dynamique :", v["W3"]["verdict"], v["W3"]["T_star_reel"], v["W3"]["T_star_melange"])
assert v["W3"]["verdict"] != "PIC EXPLIQUÉ PAR LA DISPERSION"
# Dispersion pure : valeurs iid, écart-type culminant à 1 -> pic expliqué par la dispersion
res = make(lambda i, t: rng.normal(size=(P, L)) * (0.2 + np.exp(-(t - 1.0) ** 2 / 0.05)), lambda i, t: rng.normal(size=(P, L)))
v = ns["analyse_v2"](res, True); print("W3 dispersion :", v["W3"]["verdict"])
assert v["W3"]["verdict"] == "PIC EXPLIQUÉ PAR LA DISPERSION"

# Cas 3 : W4a/W4c. Chaque prompt a son propre T_c, lié à son instabilité initiale (négativement)
tc = rng.uniform(0.7, 1.5, P)
def S_prompt(i, t):
    amp = np.exp(-(t - tc) ** 2 / 0.04)[:, None]
    x = rng.normal(size=(P, L)) * amp
    if abs(t - 1.0) < 1e-9: x[:, :16] += (2.0 - tc)[:, None] * 5
    return x
res = make(S_prompt, lambda i, t: rng.normal(size=(P, L)))
v = ns["analyse_v2"](res, True); print("W4a :", v["W4a"]["verdict"], round(v["W4a"]["rho"], 2), "| W4c :", v["W4c"]["verdict"], round(v["W4c"]["rho"], 2), "| W4b :", v["W4b"]["verdict"], v["W4b"]["ci"])
assert v["W4a"]["verdict"] == "SOUTENU" and v["W4c"]["verdict"] == "SOUTENU" and v["W4b"]["verdict"] != "SOUTENU"
# Sans individualité : W4a non soutenu
res = make(lambda i, t: rng.normal(size=(P, L)) * np.exp(-(t - 1.0) ** 2 / 0.04), lambda i, t: rng.normal(size=(P, L)))
v = ns["analyse_v2"](res, True); print("W4a nul :", v["W4a"]["verdict"], round(v["W4a"]["rho"], 2), v["W4a"]["p_perm"])
assert v["W4a"]["verdict"] != "SOUTENU"
# Dérive : T_c du début 0,9, de la fin 1,3 -> W4b SOUTENU
def S_drift(i, t):
    k = np.arange(L)[None, :]; c = 0.9 + 0.4 * k / L
    return rng.normal(size=(P, L)) * np.exp(-(t - c) ** 2 / 0.04)
res = make(S_drift, lambda i, t: rng.normal(size=(P, L)))
v = ns["analyse_v2"](res, True); print("W4b dérive :", v["W4b"]["verdict"], round(v["W4b"]["diff_moyenne"], 3))
assert v["W4b"]["verdict"] == "SOUTENU" and v["W4b"]["diff_moyenne"] > 0
print("TESTS V2 PASSENT")

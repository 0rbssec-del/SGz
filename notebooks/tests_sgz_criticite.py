"""Tests unitaires du notebook SGZ criticité (harnais local, CPU, sans Internet).
Exécute les cellules de configuration, prompts, mesures et témoin positif, puis teste les fonctions."""
import os
import sys

os.environ["SMOKE_TEST"] = "1"
os.environ.setdefault("OUT", "/tmp/sgz_tests_out")
sys.path.insert(0, os.path.dirname(__file__))
from build_notebook import cells_from_script

import numpy as np
import torch

cells = [c for c in cells_from_script(os.path.join(os.path.dirname(__file__), "sgz_criticite_temperature.py"))
         if c.cell_type == "code"]
ns = {}
for c in cells[:5]:          # config, prompts, mesures, témoin positif, chargement
    exec(c.source, ns)
print("cellules 1 à 5 exécutées")

# TwoNN sur dimensions connues
rng = np.random.default_rng(0)
for d_true in (2, 5):
    Z = rng.normal(size=(2000, d_true)) @ rng.normal(size=(d_true, 50))
    d, dup = ns["twonn"](torch.tensor(Z, dtype=torch.float32))
    print(f"TwoNN : dimension vraie {d_true}, estimée {d:.2f}")
    assert abs(d - d_true) < 0.15 * d_true + 0.3
Zd = np.repeat(rng.normal(size=(500, 5)), 2, axis=0)
d, dup = ns["twonn"](torch.tensor(Zd, dtype=torch.float32))
print(f"TwoNN avec doublons : fraction dupliquée {dup:.2f}, dimension {d}")
assert abs(dup - 0.5) < 1e-9 and abs(d - 5) < 1.0, (dup, d)

# Entropie avec termes p = 0 (logits -inf)
z = torch.tensor([[0.0, 0.0, -float("inf"), -float("inf")]])
e = ns["entropies_from_logits"](z, ns["Q_LIST"])
assert abs(e[1.0].item() - np.log(2)) < 1e-12 and all(torch.isfinite(v).all() for v in e.values())
print("entropie avec p = 0 : finie et exacte")

# Monotonie du niveau de S_q en T (P1, invariant attendu)
zz = torch.randn(32, 1000) * 3
for q in ns["Q_LIST"]:
    lv = [ns["entropies_from_logits"](zz / T, [q])[q].mean().item() for T in ns["T_GRID"]]
    assert np.all(np.diff(lv) > 0), q
print("niveau de S_q strictement croissant en T pour tous les q (logits aléatoires)")

# Remplissage à gauche + cache : mêmes logits que le prompt seul
model, tok, mode = ns["load_model"]("smoke-qwen3-random")
model = model.float()
a = torch.tensor([[5, 6, 7, 8, 9]])
b = torch.tensor([[0, 0, 11, 12, 13]])
attn_b = torch.tensor([[0, 0, 1, 1, 1]])
ids = torch.cat([a, b])
attn = torch.cat([torch.ones(1, 5, dtype=torch.long), attn_b])
pos = (attn.cumsum(-1) - 1).clamp(min=0)
with torch.no_grad():
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True)
    alone = model(input_ids=torch.tensor([[11, 12, 13]]), use_cache=True)
    d0 = (out.logits[1, -1] - alone.logits[0, -1]).abs().max().item()
    nxt = torch.tensor([[20], [21]])
    attn2 = torch.cat([attn, torch.ones(2, 1, dtype=torch.long)], 1)
    out2 = model(input_ids=nxt, attention_mask=attn2, position_ids=pos[:, -1:] + 1, past_key_values=out.past_key_values, use_cache=True)
    alone2 = model(input_ids=torch.tensor([[21]]), past_key_values=alone.past_key_values, use_cache=True)
    d1 = (out2.logits[1, -1] - alone2.logits[0, -1]).abs().max().item()
print(f"remplissage à gauche : écart des logits {d0:.2e} (prompt), {d1:.2e} (pas suivant avec cache)")
assert d0 < 1e-4 and d1 < 1e-4

# Bruit de Gumbel : mêmes graines -> mêmes nombres (appariement entre températures)
g1 = ns["gumbel_noise"](100, [1, 2], "cpu")
g2 = ns["gumbel_noise"](100, [1, 2], "cpu")
assert torch.equal(g1, g2) and not torch.equal(g1[0], g1[1])
print("Gumbel : appariement exact, lignes indépendantes")

# Verdicts : cas connus
pv = ns["peak_verdict"]
grid = ns["T_GRID"]
assert pv(np.eye(12)[5], np.full(100, 1.0), grid, (0.8, 1.2))["verdict"] == "SOUTENU"
assert pv(np.eye(12)[9], np.full(100, 1.6), grid, (0.8, 1.2))["verdict"] == "RÉFUTÉ"
assert pv(np.eye(12)[0], np.full(100, 0.2), grid, (0.8, 1.2))["interior"] is False
mix = np.r_[np.full(50, 1.0), np.full(50, 1.6)]
assert pv(np.eye(12)[5], mix, grid, (0.8, 1.2))["verdict"] == "INDÉTERMINÉ"
assert pv(np.eye(12)[5], np.full(100, 1.0), grid, (0.8, 1.2), interpretable=False)["verdict"] == "NON INTERPRÉTABLE"
print("verdicts : SOUTENU, RÉFUTÉ, bord, INDÉTERMINÉ, NON INTERPRÉTABLE corrects")
assert ns["POSITIVE_CONTROL_OK"]
print("TOUS LES TESTS PASSENT")

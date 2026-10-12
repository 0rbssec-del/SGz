"""phi6 reagit-il a (T, top-p) ? Experience discriminante, CPU seulement.

Question : la commande du PID (T, p) peut-elle deplacer la mesure d'asymetrie ?
Dans CAAAgent, phi6 est calcule sur le prompt avant echantillonnage : il ne
depend pas de (T, p) par construction (colonne phi6_statique, controle).
Ici phi6 est recalcule sur la continuation generee (phi6_gen) : boucle fermee.

phi6_gen = moyenne, sur les tokens generes, de KL(p(.|prompt+gen) || p(.|prive+prompt+gen)).

Critere de refutation (fixe avant execution) : si l'ecart de phi6_gen entre les
cellules extremes de la grille est <= 2 ecarts-types entre graines, la
commande (T, p) ne pilote pas A et la boucle PID du depot est sans objet.

Usage :
    python experiments/phi6_vs_sampling.py --model gpt2           # ~10 min CPU
    python experiments/phi6_vs_sampling.py --smoke                # modele aleatoire, teste le code seulement
"""
import argparse
import csv
import itertools
import time

import numpy as np
import torch

from src.asymmetry.metrics import kl_logits

PAIRS = [
    ("Private note: the vault code is 4721.", "The vault code is"),
    ("Private note: Alice moved to Lisbon last month.", "These days Alice lives in"),
    ("Private note: the meeting was moved to Friday.", "The meeting will take place on"),
    ("Private note: the project is code-named Heron.", "The project's code name is"),
    ("Private note: the patient is allergic to penicillin.", "Before prescribing, remember the patient is allergic to"),
    ("Private note: our server runs on port 8443.", "To connect, use port"),
    ("Private note: the cat's name is Biscuit.", "Our cat is called"),
    ("Private note: the budget was cut to 12,000 dollars.", "The remaining budget is"),
    ("Private note: the bridge is closed for repairs.", "To cross the river today, drivers should"),
    ("Private note: Bob prefers tea, never coffee.", "When Bob visits, offer him"),
    ("Private note: the exam covers chapters 3 to 5 only.", "To prepare for the exam, students should read"),
    ("Private note: the launch was delayed until March.", "The product launch is now scheduled for"),
]
TEMPS = [0.5, 0.8, 1.0, 1.3, 1.6]
TOP_PS = [0.5, 0.8, 0.95, 0.99]


class ByteTokenizer:
    """Tokeniseur octet pour --smoke (aucun telechargement)."""
    def encode(self, text):
        return list(text.encode("utf-8"))


def load(model_name, smoke):
    if smoke:
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=256, n_positions=512, n_embd=64, n_layer=2, n_head=2)
        return GPT2LMHeadModel(cfg).eval(), ByteTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    return model, tok


@torch.no_grad()
def next_logits(model, ids):
    return model(torch.tensor([ids])).logits[0, -1, :]


def sample_top_p(logits, T, p, gen):
    # Meme regle que LLMWrapper.sample.
    probs = torch.softmax(logits / T, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    mask = torch.cumsum(sorted_probs, dim=-1) <= p
    mask[0] = True
    sorted_probs = sorted_probs * mask
    idx = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=gen)
    return int(sorted_idx[idx])


@torch.no_grad()
def phi6_gen(model, priv_ids, prompt_ids, gen_ids):
    """KL brute moyenne sur les positions generees (sans tanh)."""
    sans = prompt_ids + gen_ids
    avec = priv_ids + prompt_ids + gen_ids
    n = len(gen_ids)
    lg_sans = model(torch.tensor([sans])).logits[0, len(prompt_ids) - 1:len(sans) - 1]
    lg_avec = model(torch.tensor([avec])).logits[0, len(priv_ids) + len(prompt_ids) - 1:len(avec) - 1]
    assert lg_sans.shape[0] == lg_avec.shape[0] == n
    return float(kl_logits(lg_sans, lg_avec))


@torch.no_grad()
def run(model, tok, n_tokens, seeds, out_path):
    rows = []
    t0 = time.time()
    for k, (priv, prompt) in enumerate(PAIRS):
        priv_ids = tok.encode(priv + "\n\n")
        prompt_ids = tok.encode(prompt)
        static = float(kl_logits(next_logits(model, prompt_ids), next_logits(model, priv_ids + prompt_ids)))
        for T, p, s in itertools.product(TEMPS, TOP_PS, range(seeds)):
            gen = torch.Generator().manual_seed(1000 * k + s)
            ids = priv_ids + prompt_ids
            gen_ids = []
            for _ in range(n_tokens):
                tok_id = sample_top_p(next_logits(model, ids), T, p, gen)
                ids.append(tok_id)
                gen_ids.append(tok_id)
            kl = phi6_gen(model, priv_ids, prompt_ids, gen_ids)
            rows.append({"pair": k, "T": T, "top_p": p, "seed": s,
                         "phi6_statique_kl": static, "phi6_gen_kl": kl,
                         "phi6_gen_tanh": float(np.tanh(kl))})
        print(f"paire {k + 1}/{len(PAIRS)}  {time.time() - t0:.0f}s", flush=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows, seeds):
    kl = {}
    for r in rows:
        kl.setdefault((r["T"], r["top_p"]), {}).setdefault(r["seed"], []).append(r["phi6_gen_kl"])
    print("\nphi6_gen (KL brute, moyenne sur les paires) ; lignes = T, colonnes = top_p")
    print("T\\p   " + "  ".join(f"{p:>6}" for p in TOP_PS))
    for T in TEMPS:
        print(f"{T:<5} " + "  ".join(f"{np.mean([np.mean(v) for v in kl[(T, p)].values()]):6.3f}" for p in TOP_PS))
    # Ecart-type entre graines (moyenne sur les paires), moyenne sur les cellules.
    seed_sd = np.mean([np.std([np.mean(v) for v in cell.values()]) for cell in kl.values()]) if seeds > 1 else float("nan")
    lo = np.mean([np.mean(v) for v in kl[(TEMPS[0], TOP_PS[0])].values()])
    hi = np.mean([np.mean(v) for v in kl[(TEMPS[-1], TOP_PS[-1])].values()])
    from scipy.stats import spearmanr
    rho_T = spearmanr([r["T"] for r in rows], [r["phi6_gen_kl"] for r in rows])
    rho_p = spearmanr([r["top_p"] for r in rows], [r["phi6_gen_kl"] for r in rows])
    print(f"\nCellule (T={TEMPS[0]}, p={TOP_PS[0]}) = {lo:.3f} ; cellule (T={TEMPS[-1]}, p={TOP_PS[-1]}) = {hi:.3f}")
    print(f"Ecart extremes = {abs(hi - lo):.3f} ; ecart-type entre graines = {seed_sd:.3f}")
    print(f"Spearman(T, phi6_gen) = {rho_T[0]:.3f} (p={rho_T[1]:.2g}) ; Spearman(top_p, phi6_gen) = {rho_p[0]:.3f} (p={rho_p[1]:.2g})")
    print("phi6_statique varie-t-il avec (T, p) ? ",
          "non (constant par paire, attendu)" if len({(r['pair'], r['phi6_statique_kl']) for r in rows}) == len(PAIRS) else "OUI (anomalie)")
    verdict = "REFUTE : (T, p) ne pilote pas phi6" if abs(hi - lo) <= 2 * seed_sd else "non refute : (T, p) deplace phi6"
    print(f"Verdict selon le critere pre-enregistre : {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-tokens", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/phi6_vs_sampling.csv")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    model, tok = load(a.model, a.smoke)
    torch.set_num_threads(os.cpu_count() or 1)
    rows = run(model, tok, a.n_tokens, a.seeds, a.out)
    if a.smoke:
        print("\n[--smoke] modele aleatoire : chiffres sans valeur, seul le code est teste.")
    summarize(rows, a.seeds)

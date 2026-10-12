"""Composantes phi1..phi6."""
import torch
import torch.nn.functional as F
import numpy as np


def kl_logits(logits_p, logits_q, temperature=1.0):
    """KL(p || q) sommee sur le vocabulaire, moyennee sur les autres dimensions.

    Remplace F.kl_div(..., reduction="batchmean"), qui divise par size(0) :
    sur un vecteur 1D de logits, c'est la taille du vocabulaire.
    """
    logp = F.log_softmax(logits_p / temperature, dim=-1)
    logq = F.log_softmax(logits_q / temperature, dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(dim=-1)
    return kl.mean()


def _perplexity(model, prompts, device):
    total_loss, total_tokens = 0.0, 0
    for p in prompts:
        inputs = model.tokenize(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        total_loss += out.loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)
    return float(np.exp(total_loss / max(total_tokens, 1)))


def phi1_perplexity_asymmetry(model, prompts_public, prompts_private, device="cpu"):
    pp_pub = _perplexity(model, prompts_public, device)
    pp_priv = _perplexity(model, prompts_private, device)
    return abs(pp_priv - pp_pub) / (pp_priv + pp_pub + 1e-8)


def phi2_kl_libre_aligne(logits_libre, logits_aligne, temperature=1.0):
    kl = kl_logits(logits_libre, logits_aligne, temperature)
    return float(torch.tanh(kl).item())


def phi3_information_gradient(attention, context_mask):
    coverage = attention.mean(dim=1).sum(dim=-2)
    used = (coverage * context_mask).sum() / (context_mask.sum() + 1e-8)
    total = coverage.sum() / (coverage.numel() + 1e-8)
    return float((1.0 - used / (total + 1e-8)).clamp(0, 1).item())


def phi4_head_asymmetry(attention):
    A_mean = attention.mean(dim=1, keepdim=True)
    diff = (attention - A_mean).flatten(2).norm(dim=-1)
    return float(torch.tanh(diff.mean()).item())


def _get_logits(model, prompt, device):
    inputs = model.tokenize(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :]


def phi5_paraphrase_asymmetry(model, prompts, paraphrases, device="cpu"):
    kls = []
    for p, p2 in zip(prompts, paraphrases):
        logits1 = _get_logits(model, p, device)
        logits2 = _get_logits(model, p2, device)
        kl = kl_logits(logits1, logits2)
        kls.append(float(kl.item()))
    return float(torch.tanh(torch.tensor(np.mean(kls))).item())


def phi6_private_asymmetry(model, prompts, private_info, device="cpu"):
    kls = []
    for p in prompts:
        logits_sans = _get_logits(model, p, device)
        logits_avec = _get_logits(model, f"{private_info}\n\n{p}", device)
        kl = kl_logits(logits_sans, logits_avec)
        kls.append(float(kl.item()))
    return float(torch.tanh(torch.tensor(np.mean(kls))).item())


def compute_A(model, prompts_public, prompts_private, paraphrases, private_info,
              logits_libre=None, logits_aligne=None, attention=None,
              device="cpu", weights=None):
    if weights is None:
        weights = {"phi1": 1, "phi2": 2, "phi3": 1, "phi4": 1, "phi5": 1, "phi6": 2}
    phi1 = phi1_perplexity_asymmetry(model, prompts_public, prompts_private, device)
    phi2 = phi2_kl_libre_aligne(logits_libre, logits_aligne) if logits_libre is not None else 0.0
    phi3 = phi3_information_gradient(attention, torch.ones_like(attention[:, 0, 0, :])) if attention is not None else 0.0
    phi4 = phi4_head_asymmetry(attention) if attention is not None else 0.0
    phi5 = phi5_paraphrase_asymmetry(model, prompts_public, paraphrases, device)
    phi6 = phi6_private_asymmetry(model, prompts_public, private_info, device)
    phis = {"phi1": phi1, "phi2": phi2, "phi3": phi3, "phi4": phi4, "phi5": phi5, "phi6": phi6}
    Z = sum(weights.values())
    A = sum(weights[k] * phis[k] for k in phis) / Z
    return {"A": A, **phis}

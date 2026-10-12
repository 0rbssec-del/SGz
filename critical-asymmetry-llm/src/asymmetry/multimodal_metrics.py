"""Metriques multi-modales phi7..phi9."""
import torch
import torch.nn.functional as F
import numpy as np
from .metrics import kl_logits


def phi7_intermodal_kl(logits_sans_image, logits_avec_image, temperature=1.0):
    kl = kl_logits(logits_sans_image, logits_avec_image, temperature)
    return float(torch.tanh(kl).item())


def phi8_visual_salience(attn_vt):
    salience = attn_vt.mean(dim=-1)
    salience = salience / (salience.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(salience * (salience + 1e-8).log()).sum(dim=-1)
    max_entropy = np.log(salience.size(-1))
    concentration = 1.0 - (entropy / max_entropy)
    return float(concentration.mean().clamp(0, 1).item())


def phi9_private_visual_kl(logits_pub, logits_priv, temperature=1.0):
    kl = kl_logits(logits_pub, logits_priv, temperature)
    return float(torch.tanh(kl).item())


def compute_A_multimodal(phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, weights=None):
    if weights is None:
        weights = {"phi1": 1.0, "phi2": 2.0, "phi3": 1.0, "phi4": 1.0, "phi5": 1.0,
                   "phi6": 2.0, "phi7": 2.0, "phi8": 1.0, "phi9": 2.0}
    phis = {"phi1": phi1, "phi2": phi2, "phi3": phi3, "phi4": phi4, "phi5": phi5,
            "phi6": phi6, "phi7": phi7, "phi8": phi8, "phi9": phi9}
    Z = sum(weights.values())
    A = sum(weights[k] * phis[k] for k in phis) / Z
    return {"A_MM": A, **phis}

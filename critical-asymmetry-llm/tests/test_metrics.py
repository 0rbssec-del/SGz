import torch
from src.asymmetry.metrics import phi2_kl_libre_aligne, phi4_head_asymmetry


def test_phi2_identical():
    logits = torch.randn(4, 100)
    assert phi2_kl_libre_aligne(logits, logits) < 1e-5


def test_phi2_different():
    a = torch.randn(4, 100)
    b = torch.randn(4, 100) * 3
    assert phi2_kl_libre_aligne(a, b) > 0.1


def test_phi4_symmetric_heads():
    attn = torch.ones(2, 8, 10, 10) / 10
    assert phi4_head_asymmetry(attn) < 1e-3


def test_phi4_asymmetric_heads():
    attn = torch.randn(2, 8, 10, 10).softmax(dim=-1)
    assert phi4_head_asymmetry(attn) > 0.05


def test_phi2_1d_logits_not_divided_by_vocab():
    # Vecteur 1D comme model.logits() : la KL doit etre sommee sur le vocabulaire.
    torch.manual_seed(0)
    a, b = torch.randn(32000) * 3, torch.randn(32000) * 3
    assert phi2_kl_libre_aligne(a, b) > 0.9

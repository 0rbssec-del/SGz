import torch
from src.asymmetry.multimodal_metrics import (
    phi7_intermodal_kl, phi8_visual_salience, phi9_private_visual_kl,
)


def test_phi7_identical():
    x = torch.randn(2, 100)
    assert phi7_intermodal_kl(x, x) < 1e-5


def test_phi7_different():
    a = torch.randn(2, 100)
    b = torch.randn(2, 100) * 2
    assert phi7_intermodal_kl(a, b) > 0.1


def test_phi8_uniform():
    attn = torch.ones(2, 10, 5) / 5
    assert phi8_visual_salience(attn) < 0.2


def test_phi9_different():
    a = torch.randn(2, 100)
    b = torch.randn(2, 100) * 3
    assert phi9_private_visual_kl(a, b) > 0.1

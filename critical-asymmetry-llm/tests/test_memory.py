import torch
from src.memory.private_memory import PrivateMemory


def test_memory_update():
    m = PrivateMemory(dim=8)
    m.update(surprise=1.0)
    assert m.a.norm().item() > 0


def test_memory_decay():
    m = PrivateMemory(dim=8, mu=0.1, nu=0.0)
    m.a = torch.ones(8)
    m.update(surprise=0.0)
    assert m.a.norm().item() < 8 ** 0.5

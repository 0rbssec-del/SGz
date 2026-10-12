import torch
import torch.nn as nn


class PrivateMemory(nn.Module):
    def __init__(self, dim=768, mu=0.01, nu=0.1, rho=0.0):
        super().__init__()
        self.dim = dim
        self.mu = mu
        self.nu = nu
        self.rho = rho
        self.register_buffer("a", torch.zeros(dim))

    @torch.no_grad()
    def update(self, surprise, neighbor_a=None):
        delta = -self.mu * self.a + self.nu * surprise
        if neighbor_a is not None and self.rho > 0:
            delta += self.rho * (neighbor_a - self.a)
        self.a += delta

    def inject(self, context_embedding):
        return context_embedding + self.a

    def asymmetry_norm(self):
        return float(self.a.norm().item())

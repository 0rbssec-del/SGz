import numpy as np
import torch


class MultiModalMultiAgent:
    def __init__(self, agents, W=None, K_tv=None,
                 alpha=0.1, gamma=0.05, rho=0.0,
                 kappa_tv=0.1, kappa_vt=0.08):
        self.agents = agents
        self.N = len(agents)
        self.alpha, self.gamma, self.rho = alpha, gamma, rho
        self.kappa_tv, self.kappa_vt = kappa_tv, kappa_vt
        self.W = W if W is not None else self._random_W()
        self.K_tv = K_tv if K_tv is not None else np.eye(self.N) * kappa_tv
        self.A_t = np.array([0.5] * self.N)
        self.A_v = np.array([0.5] * self.N)
        self.A_star = np.array([0.5] * self.N)

    def _random_W(self, p=0.3):
        W = (np.random.rand(self.N, self.N) < p).astype(float)
        W *= np.random.rand(self.N, self.N)
        np.fill_diagonal(W, 0)
        return W

    @torch.no_grad()
    def step(self, batch_inputs):
        results = []
        for i, agent in enumerate(self.agents):
            r = agent.step(**batch_inputs[i], tache="general")
            results.append(r)
            self.A_t[i] = r["A_lisse"]
            self.A_v[i] = r["phis"].get("phi8", 0.5)
        A_t_new = self.A_t.copy()
        A_v_new = self.A_v.copy()
        for i in range(self.N):
            coupling_agents = sum(self.W[i, j] * np.tanh(self.A_t[j] - self.A_t[i])
                                  for j in range(self.N))
            coupling_mod = self.kappa_tv * (self.A_v[i] - self.A_t[i])
            A_t_new[i] += (self.alpha * (self.A_star[i] - self.A_t[i])
                           + coupling_agents + coupling_mod)
            A_v_new[i] += self.kappa_vt * (self.A_t[i] - self.A_v[i])
        self.A_t = np.clip(A_t_new, 0.0, 1.0)
        self.A_v = np.clip(A_v_new, 0.0, 1.0)
        for i in range(self.N):
            social = sum(self.W[i, j] * (self.A_star[j] - self.A_star[i])
                         for j in range(self.N))
            self.A_star[i] += self.gamma * social
        return {"A_t": self.A_t.copy(), "A_v": self.A_v.copy(),
                "A_star": self.A_star.copy(),
                "A_MM": (self.A_t + self.A_v) / 2, "results": results}

    def order_parameter(self):
        return {"m_t": float(np.mean(self.A_t) - 0.5),
                "m_v": float(np.mean(self.A_v) - 0.5),
                "m_MM": float(np.mean((self.A_t + self.A_v) / 2) - 0.5)}

    def critical_fraction(self, lo=0.4, hi=0.6):
        return {"crit_t": float(np.mean((self.A_t >= lo) & (self.A_t <= hi))),
                "crit_v": float(np.mean((self.A_v >= lo) & (self.A_v <= hi)))}

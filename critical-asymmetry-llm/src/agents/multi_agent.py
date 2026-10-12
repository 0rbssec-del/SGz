import numpy as np
import torch


class MultiAgentCAA:
    def __init__(self, agents, W=None, alpha=0.1, gamma=0.05, rho=0.0):
        self.agents = agents
        self.N = len(agents)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.W = W if W is not None else self._random_asymmetric_W()
        self.A = np.array([0.5] * self.N)
        self.A_star = np.array([0.5] * self.N)

    def _random_asymmetric_W(self, p=0.3):
        W = (np.random.rand(self.N, self.N) < p).astype(float)
        W *= np.random.rand(self.N, self.N)
        np.fill_diagonal(W, 0)
        return W

    def step(self, prompts, private_infos, taches):
        results = []
        for i, agent in enumerate(self.agents):
            r = agent.step(prompts[i], private_infos[i], taches[i])
            results.append(r)
            self.A[i] = r["A_lisse"]
        A_new = self.A.copy()
        for i in range(self.N):
            coupling = sum(self.W[i, j] * np.tanh(self.A[j] - self.A[i])
                           for j in range(self.N))
            A_new[i] += self.alpha * (self.A_star[i] - self.A[i]) + coupling
        self.A = np.clip(A_new, 0.0, 1.0)
        for i in range(self.N):
            social = sum(self.W[i, j] * (self.A_star[j] - self.A_star[i])
                         for j in range(self.N))
            self.A_star[i] += self.gamma * social
        for i, agent in enumerate(self.agents):
            neighbor_a = None
            if self.rho > 0:
                vecs = [self.agents[j].memory.a for j in range(self.N) if self.W[i, j] > 0]
                if vecs:
                    neighbor_a = torch.stack(vecs).mean(dim=0)
            agent.feedback(results[i]["surprise"], neighbor_a)
        return {"A": self.A.copy(), "A_star": self.A_star.copy(), "results": results}

    def order_parameter(self):
        return float(np.mean(self.A) - 0.5)

    def critical_fraction(self, lo=0.4, hi=0.6):
        return float(np.mean((self.A >= lo) & (self.A <= hi)))

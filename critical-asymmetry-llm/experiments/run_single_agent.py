import numpy as np
from src.agents.caa_agent import CAAAgent


def run(steps=500, A_star=0.5):
    agent = CAAAgent(model=None, A_star=A_star)
    history = []
    for t in range(steps):
        r = agent.step(f"prompt_{t}", f"private_{t}")
        history.append(r["A_lisse"])
    return np.array(history)


if __name__ == "__main__":
    h = run()
    print(f"A moyen = {h.mean():.3f}, variance = {h.var():.4f}")
    print(f"Taux critique = {((h >= 0.4) & (h <= 0.6)).mean():.3f}")

import numpy as np
from src.agents.caa_agent import CAAAgent
from src.agents.multi_agent import MultiAgentCAA


def run(N=10, steps=500, alpha=0.1, gamma=0.05, rho=0.0):
    agents = [CAAAgent(model=None, agent_id=i, A_star=0.5) for i in range(N)]
    net = MultiAgentCAA(agents, alpha=alpha, gamma=gamma, rho=rho)
    history_A, history_m, history_crit = [], [], []
    for t in range(steps):
        prompts = [f"prompt_{i}_{t}" for i in range(N)]
        privates = [f"private_{i}" for i in range(N)]
        taches = ["general"] * N
        out = net.step(prompts, privates, taches)
        history_A.append(out["A"].copy())
        history_m.append(net.order_parameter())
        history_crit.append(net.critical_fraction())
    return {"A": np.array(history_A), "m": np.array(history_m),
            "crit": np.array(history_crit), "W": net.W}


if __name__ == "__main__":
    results = run(N=10, steps=500)
    print(f"m final = {results['m'][-1]:.3f}")
    print(f"Fraction critique finale = {results['crit'][-1]:.3f}")

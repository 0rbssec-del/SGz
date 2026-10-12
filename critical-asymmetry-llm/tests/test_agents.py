from src.agents.caa_agent import CAAAgent


def test_caa_agent_step():
    agent = CAAAgent(model=None, A_star=0.5)
    r = agent.step("prompt", "private")
    assert "A_lisse" in r

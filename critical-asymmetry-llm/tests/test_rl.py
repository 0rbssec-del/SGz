from src.rl.rewards import (
    reward_A, reward_repression, reward_chaos, reward_total,
)


def test_reward_A_max():
    assert reward_A(0.5, 0.5) == 1.0


def test_reward_A_decays():
    assert reward_A(0.5, 0.5) > reward_A(0.7, 0.5)


def test_reward_repression():
    assert reward_repression(0.2) < 0


def test_reward_chaos():
    assert reward_chaos(0.9) < 0


def test_reward_total():
    r = reward_total(0.8, 0.5, 0.3)
    assert r > 0

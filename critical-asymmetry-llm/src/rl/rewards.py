import math


def reward_tache(score): return float(score)
def reward_A(A_hat, A_star=0.5, sigma=0.1):
    return math.exp(-((A_hat - A_star) ** 2) / (2 * sigma ** 2))
def reward_prive(kl_prive): return float(kl_prive)
def reward_repression(A_hat, seuil=0.3): return -0.5 * max(0.0, seuil - A_hat)
def reward_chaos(A_hat, seuil=0.7): return -0.5 * max(0.0, A_hat - seuil)

def reward_total(score_tache, A_hat, kl_prive, A_star=0.5, lam=0.5, mu=0.3):
    return (reward_tache(score_tache) + lam * reward_A(A_hat, A_star)
            + mu * reward_prive(kl_prive) + reward_repression(A_hat)
            + reward_chaos(A_hat))

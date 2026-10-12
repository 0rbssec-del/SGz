import math
import numpy as np


class MultiModalController:
    def __init__(self, T0=1.0, p0=0.85, lambda_v0=0.5, tau_v0=1.0,
                 alpha=0.12, beta=0.35, gamma=0.20, delta=0.10, A0=0.25,
                 T_range=(0.5, 1.5), p_range=(0.5, 0.99),
                 lambda_v_range=(0.0, 1.0), tau_v_range=(0.5, 1.5)):
        self.T = T0
        self.p = p0
        self.lambda_v = lambda_v0
        self.tau_v = tau_v0
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.A0 = A0
        self.T_range = T_range
        self.p_range = p_range
        self.lambda_v_range = lambda_v_range
        self.tau_v_range = tau_v_range

    def predict_A(self):
        return (self.A0 + self.alpha * math.log(self.T)
                + self.beta * (1 - self.p) + self.gamma * self.lambda_v
                + self.delta * math.log(self.tau_v))

    def update(self, delta_u):
        self.lambda_v = float(np.clip(self.lambda_v + 0.5 * delta_u / self.gamma, *self.lambda_v_range))
        self.p = float(np.clip(self.p - 0.3 * delta_u / self.beta, *self.p_range))
        self.T = float(np.clip(self.T * math.exp(0.15 * delta_u / self.alpha), *self.T_range))
        self.tau_v = float(np.clip(self.tau_v * math.exp(0.05 * delta_u / self.delta), *self.tau_v_range))
        return self.T, self.p, self.lambda_v, self.tau_v

    def state(self):
        return {"T": self.T, "p": self.p, "lambda_v": self.lambda_v,
                "tau_v": self.tau_v, "A_pred": self.predict_A()}

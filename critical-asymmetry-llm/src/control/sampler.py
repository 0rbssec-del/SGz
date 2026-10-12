import math
import numpy as np


class TemperatureTopPController:
    def __init__(self, T0=1.0, p0=0.85, alpha=0.15, beta=0.4, A0=0.3):
        self.T = T0
        self.p = p0
        self.alpha = alpha
        self.beta = beta
        self.A0 = A0

    def u_to_Tp(self, u):
        T = self.T
        p = 1.0 - (u - self.alpha * math.log(T)) / self.beta
        T = float(np.clip(T, 0.5, 1.5))
        p = float(np.clip(p, 0.5, 0.99))
        return T, p

    def update(self, delta_u):
        u = self.alpha * math.log(self.T) + self.beta * (1 - self.p)
        u_new = u + delta_u
        self.T, self.p = self.u_to_Tp(u_new)
        return self.T, self.p

    def predict_A(self):
        return self.A0 + self.alpha * math.log(self.T) + self.beta * (1 - self.p)

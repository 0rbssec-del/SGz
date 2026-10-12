import numpy as np
from collections import deque


class PIDController:
    def __init__(self, Kp=0.3, Ki=0.05, Kd=0.1, A_star=0.5):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.A_star = A_star
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, A_hat):
        error = self.A_star - A_hat
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class SlidingWindowA:
    def __init__(self, window=16):
        self.window = deque(maxlen=window)

    def update(self, A):
        self.window.append(A)
        return float(np.mean(self.window))

    def variance(self):
        return float(np.var(self.window)) if len(self.window) > 1 else 0.0

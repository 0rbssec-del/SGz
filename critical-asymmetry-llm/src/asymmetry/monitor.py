from collections import deque
import numpy as np


class AsymmetryMonitor:
    def __init__(self, window=16, A_star=0.5):
        self.window = deque(maxlen=window)
        self.A_star = A_star

    def update(self, A):
        self.window.append(A)
        return float(np.mean(self.window))

    def variance(self):
        return float(np.var(self.window)) if len(self.window) > 1 else 0.0

    def is_critical(self, lo=0.4, hi=0.6):
        return lo <= np.mean(self.window) <= hi

    def gradient(self):
        if len(self.window) < 2:
            return 0.0
        return self.window[-1] - self.window[-2]

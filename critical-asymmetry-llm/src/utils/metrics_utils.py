import numpy as np


def moving_average(x, window=16):
    if len(x) < window:
        return np.mean(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def critical_fraction(A_values, lo=0.4, hi=0.6):
    A_values = np.asarray(A_values)
    return float(np.mean((A_values >= lo) & (A_values <= hi)))

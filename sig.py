import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_prime(x):
    s = sig(x)
    return s * (1 - s)

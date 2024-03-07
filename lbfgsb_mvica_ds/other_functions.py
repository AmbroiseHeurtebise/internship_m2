import numpy as np


class Memory_callback():
    def __init__(self, m, p, dilation_scale, shift_scale):
        self.memory_W = []
        self.memory_A = []
        self.memory_B = []
        self.m = m
        self.p = p
        self.dilation_scale = dilation_scale
        self.shift_scale = shift_scale

    def __call__(self, W_A_B):
        m = self.m
        p = self.p
        self.memory_W.append(W_A_B[:m*p**2].reshape((m, p, p)))
        self.memory_A.append(W_A_B[m*p**2: m*p*(p+1)].reshape((m, p)) / self.dilation_scale)
        self.memory_B.append(W_A_B[m*p*(p+1):].reshape((m, p)) / self.shift_scale)


def compute_lambda(s, n_concat=1):
    t = np.linspace(0, 1, len(s) // n_concat)
    lambdas = np.zeros(n_concat)
    for i, s_ in enumerate(np.hsplit(s, n_concat)):
        lambdas[i] = np.sqrt(np.mean(np.gradient(s_) ** 2) / np.mean(np.gradient(s_) ** 2 * t ** 2))
    return np.mean(lambdas)

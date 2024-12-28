import numpy as np


class Memory_callback():
    def __init__(self, m, p, dilation_scale, shift_scale):
        self.memory_W = []
        self.memory_dilations = []
        self.memory_shifts = []
        self.m = m
        self.p = p
        self.dilation_scale = dilation_scale
        self.shift_scale = shift_scale

    def __call__(self, W_dilations_shifts):
        m = self.m
        p = self.p
        self.memory_W.append(W_dilations_shifts[:m*p**2].reshape((m, p, p)))
        self.memory_dilations.append(W_dilations_shifts[m*p**2: m*p*(p+1)].reshape((m, p)) / self.dilation_scale)
        self.memory_shifts.append(W_dilations_shifts[m*p*(p+1):].reshape((m, p)) / self.shift_scale)


def compute_lambda(s, n_concat=1):
    t = np.linspace(0, 1, len(s) // n_concat)
    lambdas = np.zeros(n_concat)
    for i, s_ in enumerate(np.hsplit(s, n_concat)):
        lambdas[i] = np.sqrt(np.mean(np.gradient(s_) ** 2) / np.mean(np.gradient(s_) ** 2 * t ** 2))
    return np.mean(lambdas)


def compute_dilation_shift_scales(
    max_dilation, max_shift, W_scale, dilation_scale_per_source, S_avg, n_concat, m,
):
    if max_shift > 0:
        shift_scale = W_scale / max_shift  # scalar
    else:
        shift_scale = 1.
    if max_dilation > 1:
        dilation_scale = W_scale / (max_dilation - 1)  # scalar
        if dilation_scale_per_source:
            lambdas = np.array([compute_lambda(s, n_concat=n_concat) for s in S_avg])
            # lambdas = 1 / lambdas
            dilation_scale *= lambdas  # vector of length p
            dilation_scale = np.array([dilation_scale] * m)  # matrix of shape (m, p)
    else:
        dilation_scale = 1.
    return dilation_scale, shift_scale

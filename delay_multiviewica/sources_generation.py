import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from delay_multiviewica import _apply_delay, _apply_delay_one_sub


def _create_sources(n_sub, p, n, delay_max=None, noise_sources=0.05, random_state=None):
    rng = check_random_state(random_state)
    if delay_max is None:
        delay_max = int(0.2 * n)
    n_inner = n - delay_max
    window = np.hamming(n_inner)
    S = window * np.outer(np.ones(p),
                          np.sin(np.linspace(0, 4 * np.pi, n_inner)))
    flip = np.outer(rng.randint(0, 2, p), np.ones(n_inner)) * 2 - 1
    S *= flip
    noise_list = noise_sources * rng.randn(n_sub, p, n_inner)
    S_list = np.array([S + n for n in noise_list])
    S_list = np.concatenate(
        (S_list, np.zeros((n_sub, p, n - n_inner))), axis=2)
    tau_list = rng.randint(0, delay_max, size=n_sub)
    S_list = _apply_delay(S_list, -tau_list)
    A_list = rng.randn(n_sub, p, p)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list


def create_sources_pierre(m, p, n, delay_max, sigma=0.05, random_state=None):
    rng = check_random_state(random_state)
    delays = rng.randint(0, 1 + delay_max, m)
    s1 = np.zeros(n)
    s1[:n//2] = np.sin(np.linspace(0, np.pi, n//2))
    s2 = rng.randn(n) / 10
    S = np.c_[s1, s2].T
    N = sigma * rng.randn(m, p, n)
    A_list = rng.randn(m, p, p)
    X_list = np.array([np.dot(A, _apply_delay_one_sub(S, delay) + noise)
                      for A, noise, delay in zip(A_list, N, delays)])
    return X_list, A_list, delays
# Only works for p=2


def create_model(m, p, n, delay=None, noise=0.05, random_state=None):
    rng = check_random_state(random_state)
    if delay is None:
        delay = int(0.2 * n)
    s = np.zeros(n)
    s[:n//4] = np.sin(np.linspace(0, np.pi, n//4))
    delay_sources = rng.randint(0, n, p)
    S = np.array([np.roll(s, -delay_sources[i]) for i in range(p)])
    noise_list = noise * rng.randn(m, p, n)
    S_list = np.array([S + N for N in noise_list])
    tau_list = rng.randint(0, delay + 1, size=m)
    S_list = _apply_delay(S_list, -tau_list)
    A_list = rng.randn(m, p, p)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list, S


def _plot_delayed_sources(S_list):
    n_sub, p, n = S_list.shape
    fig, _ = plt.subplots(n_sub//2, 2)
    fig.suptitle("Delayed sources of the {} subjects".format(n_sub))
    for i in range(n_sub):
        plt.subplot(n_sub//2, 2, i+1)
        for j in range(p):
            plt.plot(S_list[i, j, :] + 3 * j)
            plt.hlines(y=3*j, xmin=0, xmax=n, linestyles='--', colors='grey')
            plt.grid()
            plt.yticks([])
            plt.title("Sources of subject {}".format(i+1))
    plt.show()

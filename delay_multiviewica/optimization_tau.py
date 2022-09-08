import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from scipy.ndimage import convolve1d


def _apply_delay_one_sub(Y, tau):
    return np.roll(Y, -tau, axis=1)


def _apply_delay(Y_list, tau_list):
    return np.array([_apply_delay_one_sub(Y, tau) for Y, tau in zip(Y_list, tau_list)])


def _create_sources(n_sub, p, n, delay_max=None, noise_sources=0.05, random_state=None):
    rng = check_random_state(random_state)
    if delay_max is None:
        delay_max = int(0.2 * n)
    n_inner = n - delay_max
    window = np.hamming(n_inner)
    S = window * np.outer(np.ones(p),
                          np.sin(np.linspace(0, 4 * math.pi, n_inner)))
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


def _loss_function(Y_list, Y_avg):
    loss = np.mean((Y_list - Y_avg) ** 2)
    return loss


def _loss_delay(S_list, tau_list):
    Y_list = _apply_delay(S_list, tau_list)
    Y_avg = np.mean(Y_list, axis=0)
    return _loss_function(Y_list, Y_avg)


def _loss_delay_ref(S_list, tau_list, Y_avg):
    Y_list = _apply_delay(S_list, tau_list)
    return _loss_function(Y_list, Y_avg)


def _delay_estimation(Y, Y_avg):
    _, n = np.shape(Y)
    conv = np.array([convolve1d(y, y_avg[::-1], mode='wrap',
                    origin=math.ceil(n/2)-1) for y, y_avg in zip(Y, Y_avg)])
    conv_norm = np.sum(conv, axis=0)
    new_tau = np.argmax(conv_norm)
    return new_tau


def distance_between_delays(tau1, tau2, n):
    assert len(tau1) == len(tau2)
    n_sub = len(tau1)
    error_total = []
    for i in range(n):
        error = 0
        tau3 = (tau2 + i) % n
        for j in range(n_sub):
            error += np.min(np.abs([tau1[j] - tau3[j] - n, tau1[j] - tau3[j], tau1[j] - tau3[j] + n]))
        error_total.append(error)
    return np.min(np.array(error_total))


def _optimization_tau_approach1(S_list, n_iter, error_tau=False, true_tau_list=None):
    n_sub, _, n = S_list.shape
    Y_avg = np.mean(S_list, axis=0)
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    gap_tau = []
    for _ in range(n_iter):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau == True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(Y_list[i], new_Y_avg)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([S_list[i]], [tau_list[i]])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau == True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau_approach2(S_list, n_iter, error_tau=False, true_tau_list=None):
    n_sub, _, n = S_list.shape
    Y_avg = S_list[0]
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    gap_tau = []
    for _ in range(n_iter):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau == True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            tau_list[i] += _delay_estimation(Y_list[i], Y_avg)
            Y_list[i] = _apply_delay([S_list[i]], [tau_list[i]])
        Y_avg = np.mean(Y_list, axis=0)
        tau_list %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau == True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau(S_list, n_iter, error_tau=False, true_tau_list=None):
    n_sub, _, n = S_list.shape
    Y_avg = S_list[0]
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    gap_tau = []
    for i in range(n_sub):
        loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
        if error_tau == True and true_tau_list is not None:
            gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        tau_list[i] = _delay_estimation(S_list[i], Y_avg)
        Y_list[i] = _apply_delay([Y_list[i]], [tau_list[i]])  # XXX
    Y_avg = np.mean(Y_list, axis=0)
    tau_list %= n
    for _ in range(n_iter-1):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau == True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(Y_list[i], new_Y_avg)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([S_list[i]], [tau_list[i]])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau == True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg

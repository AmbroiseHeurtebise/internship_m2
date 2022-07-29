import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from scipy.ndimage import convolve1d


def _apply_delay_one_sub(Y, tau):
    return np.roll(Y, -tau, axis=1)


def _apply_delay(Y_list, tau_list):
    return np.array([_apply_delay_one_sub(Y, tau) for Y, tau in zip(Y_list, tau_list)])


def _create_sources(n_sub, p, n, sources_noise, random_state=None):
    rng = check_random_state(random_state)
    n_inner = int(0.8 * n)
    window = np.hamming(n_inner)
    S = window * np.outer(np.ones(p),
                          np.sin(np.linspace(0, 4 * math.pi, n_inner)))
    flip = np.outer(rng.randint(0, 2, p), np.ones(n_inner)) * 2 - 1
    S *= flip
    noise_list = sources_noise * rng.randn(n_sub, p, n_inner)
    S_list = np.array([S + n for n in noise_list])
    S_list = np.concatenate(
        (S_list, np.zeros((n_sub, p, n - n_inner))), axis=2)
    tau_list = rng.randint(0, 0.2 * n, size=n_sub)
    S_list = _apply_delay(S_list, -tau_list)
    return S_list, tau_list


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


def _plot_delayed_sources(S_list, when):
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
    plt.savefig("figures/delayed_sources_" + when + ".pdf")
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


def _Y_avg_init(S_list, init):
    if init == 'mean':
        Y_avg = np.mean(S_list, axis=0)
    elif init == 'first_subject':
        Y_avg = S_list[0]
    else:
        raise NameError("Wrong initialization name \n")
    return Y_avg


def _new_delay_estimation(Y, Y_avg, n_sub):
    _, n = np.shape(Y)
    new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y / n_sub)
    conv = np.array([convolve1d(y, y_avg[::-1], mode='wrap',
                    origin=math.ceil(n/2)-1) for y, y_avg in zip(Y, new_Y_avg)])
    conv_norm = np.sum(conv, axis=0)
    new_tau = np.argmax(conv_norm)
    return new_tau


def _optimization_tau(S_list, n_iter, init):
    n_sub, _, n = S_list.shape
    Y_avg = _Y_avg_init(S_list, init)
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    for iter in range(n_iter):
        loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
        new_tau_list = np.zeros(n_sub, dtype=int)
        for i in range(n_sub):
            new_tau_list[i] = _new_delay_estimation(Y_list[i], Y_avg, n_sub)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([Y_list[i]], [new_tau_list[i]])
            # Y_avg = np.mean(Y_list, axis=0)  
            # it doesn't make sense to initialize Y_avg at the sources of the first subject 
            # because the formula of Y_avg changes at the second iteration
            # if we use Y_avg = np.mean(Y_list, axis=0) 
            if init == "first_subject" and iter == 0 and i > 0:
                Y_avg += Y_list[i] / i
                Y_avg *= i / (i+1)
            if init == "mean" or iter > 0:
                Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list += new_tau_list
        tau_list %= n
    return loss, tau_list, Y_avg

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from scipy.ndimage import convolve1d


def _apply_delay(Y_list, tau_list):
    return np.array([np.roll(Y, -tau, axis=1) for Y, tau in zip(Y_list, tau_list)])


def _create_sources(n_sub, p, n, sources_noise, random_state=None):
    rng = check_random_state(random_state)
    n_inner = int(0.8 * n)
    window = np.hamming(n_inner)
    S = window * np.outer(np.ones(p), np.sin(np.linspace(0, 4 * math.pi, n_inner)))
    flip = np.outer(rng.randint(0, 2, p), np.ones(n_inner)) * 2 - 1
    S *= flip
    noise_list = sources_noise * rng.randn(n_sub, p, n_inner)
    S_list = np.array([S + n for n in noise_list])
    S_list = np.concatenate((S_list, np.zeros((n_sub, p, n - n_inner))), axis=2)
    tau_list = rng.randint(0, 0.2 * n, size=n_sub)
    S_list = _apply_delay(S_list, -tau_list)
    return S_list, tau_list


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
        print("Wrong initialization name \n")
    return Y_avg


def _new_convolve(Y, Y_avg, n_sub):
    _, n = np.shape(Y)
    Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y / n_sub)
    conv = np.array([convolve1d(y, y_avg[::-1], mode='wrap',
                    origin=math.ceil(n/2)-1) for y, y_avg in zip(Y, Y_avg)])
    conv_norm = np.sum(conv, axis=0)
    new_tau = np.argmax(conv_norm)
    return new_tau


def _optimization_tau(S_list, n_iter, init):
    n_sub, _, _ = S_list.shape
    Y_avg = _Y_avg_init(S_list, init)
    Y_list = np.copy(S_list)  # _apply_delay(S_list, true_tau_list)
    tau_list = np.zeros(n_sub, dtype=int)  # true_tau_list.copy()
    loss = []
    # _plot_delayed_sources(np.array([S_list[0], Y_avg]), 'before')
    for _ in range(n_iter):
        loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
        new_tau_list = np.array([_new_convolve(Y, Y_avg, n_sub) for Y in Y_list])
        Y_list = _apply_delay(Y_list, new_tau_list)
        Y_avg = np.mean(Y_list, axis=0)
        tau_list += new_tau_list
        tau_list %= n
    return loss, tau_list


if __name__ == '__main__':
    # Parameters
    n_sub = 4  # must be even to plot the sources
    p = 3
    n = 200
    sources_noise = 0.5
    # init = 'mean'
    random_state = 4
    n_iter = 10

    # Generate data
    S_list, true_tau_list = _create_sources(n_sub, p, n, sources_noise, random_state)
    loss_true = _loss_delay(S_list, true_tau_list)

    # Optimization
    loss_mean, tau_list_mean = _optimization_tau(S_list, n_iter, 'mean')
    loss_mean = np.array(loss_mean)
    loss_first, tau_list_first = _optimization_tau(S_list, n_iter, 'first_subject')
    loss_first = np.array(loss_first)

    # Plot
    plt.semilogy(loss_mean, label='init. mean')
    plt.semilogy(loss_first, label='init. first sub.')
    plt.semilogy([loss_true] * n_iter, label='true')
    plt.title("Negative log-likelihood of the model (without function f)")
    plt.xlabel("Iterations")
    plt.ylabel("NLL")
    plt.legend()
    plt.savefig("figures/loss_optim_tau.pdf")
    plt.show()

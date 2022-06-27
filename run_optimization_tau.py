import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import correlate


def _create_sources(n_sub, p, n):
    window = np.hamming(n)
    S = window * np.outer(np.ones(p), np.sin(np.linspace(0, 4 * math.pi, n)))
    flip = np.outer(np.random.randint(0, 2, p), np.ones(n)) * 2 - 1
    S *= flip
    tau_list = np.random.randint(0, n // 4, size=n_sub)
    S_list = np.array([_delay_matrix(S, tau) for tau in tau_list])
    # sources_noise = 0.5 * np.random.randn(n_sub, p, n)
    # S_list += sources_noise
    return S_list, tau_list


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
    n_sub, p, _ = Y_list.shape
    loss = 0
    for Y in Y_list:
        Y_denoise = (n_sub * Y_avg - Y) / (n_sub - 1)
        loss += np.mean((Y - Y_denoise) ** 2) * p
    return loss


def _delay_matrix(Y, tau):
    new_Y = np.concatenate((Y[:, tau:], Y[:, :tau]), axis=1)
    return new_Y


def _optimization_tau_step(Y_list, Y_avg):
    n_sub, _, n = Y_list.shape
    new_tau_list = np.zeros(n_sub, dtype=int)
    for i in range(n_sub):
        # Cross-correlation
        corr = np.array([correlate(y, y_avg) for y, y_avg in zip(Y_list[i], Y_avg)])
        corr_norm = np.sum(corr, axis=0)
        new_tau_list[i] = np.argmax(corr_norm) - n + 1
        Y_list[i] = _delay_matrix(Y_list[i], new_tau_list[i])

    new_Y_avg = np.mean(Y_list, axis=0)
    return Y_list, new_Y_avg, new_tau_list


if __name__ == '__main__':
    # Parameters
    n_sub =4  # must be even to plot the sources
    p = 3
    n = 500

    # Generate data
    S_list, true_tau_list = _create_sources(n_sub, p, n)
    Y_list = np.copy(S_list)
    Y_avg = S_list[0]
    tau_list = np.zeros(n_sub, dtype=int)

    # _plot_delayed_sources(Y_list)

    # Optimization
    loss = []
    n_iter = 10
    for i in range(n_iter):
        loss.append(_loss_function(Y_list, Y_avg))
        Y_list, Y_avg, new_tau_list = _optimization_tau_step(Y_list, Y_avg)
        tau_list += new_tau_list
        # print(new_tau_list)

    # _plot_delayed_sources(Y_list)

    print("True tau : {}".format(true_tau_list - true_tau_list[0]))
    print("Estimated tau : {}".format(tau_list[0] - tau_list))

    # Plot
    loss = np.array(loss)
    plt.semilogy(loss)
    plt.title("Negative log-likelihood of the model (without function f)")
    plt.xlabel("Iterations")
    plt.ylabel("NLL")
    plt.savefig("figures/loss_optim_tau.pdf")
    plt.show()

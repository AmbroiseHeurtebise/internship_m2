import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import correlate


def _create_sources(p, n):
    window = np.hamming(n)
    S = np.outer(np.ones(p), np.sin(np.linspace(0, 4 * math.pi, n)))
    flip = np.outer(np.random.randint(0, 2, p), np.ones(n)) * 2 - 1
    S *= flip
    add_noise = 0.2 * np.random.randn(p, n)  # can be modified
    S = window * (S + add_noise)
    return S


def _plot_sources(S):
    for i in range(len(S)):
        plt.plot(S[i] + 3 * i)
        plt.hlines(y=3*i, xmin=0, xmax=S.shape[1], linestyles='--', colors='grey')
    plt.yticks([])
    plt.xlabel("Sample")
    plt.title("Shared sources")
    plt.show()


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


def _generate_data(n_sub, p, n):
    S = _create_sources(p, n)
    # _plot_sources(S)
    tau_list = np.random.randint(0, n // 5, size=n_sub)
    S_list = np.array([_delay_matrix(S, tau) for tau in tau_list])
    # _plot_delayed_sources(S_list)
    A_list = np.random.randn(n_sub, p, p)
    noise_model = 0.2  # can be modified
    N_list = noise_model * np.random.randn(n_sub, p, n)
    X_list = np.array([np.dot(A, S + N) for A, S, N in zip(A_list, S_list, N_list)])
    
    return X_list, A_list, S, S_list, tau_list


def _loss_function(W_list, X_list, Y_avg, noise, tau_list):
    n_sub, p, _ = W_list.shape
    loss = 0
    for _, (W, X, tau) in enumerate(zip(W_list, X_list, tau_list)):
        Y = W.dot(X)
        Y = _delay_matrix(Y, tau)
        Y_denoise = (n_sub * Y_avg - Y) / (n_sub - 1)
        loss -= np.linalg.slogdet(W)[1]
        fact = (1 - 1 / n_sub) / (2 * noise)
        loss += fact * np.mean((Y - Y_denoise) ** 2) * p  # why do we multiply by p?
    return loss


def _delay_matrix(Y, tau):
    new_Y = np.concatenate((Y[:, tau:], Y[:, :tau]), axis=1)
    return new_Y


def _optimization_tau_step(Y_list, Y_avg, tau_list):
    n_sub, p, n = Y_list.shape
    new_tau_list = np.zeros(len(tau_list), dtype=int)
    for i in range(n_sub):
        Y = _delay_matrix(Y_list[i], tau_list[i])

        # Cross_correlation
        corr = np.array([correlate(y, y_avg) for y, y_avg in zip(Y, Y_avg)])
        corr_norm = np.linalg.norm(corr, axis=0)
        new_tau_list[i] = tau_list[i] + np.argmax(corr_norm) - n + 1

    new_Y_avg = np.mean([_delay_matrix(Y, tau) for Y, tau in zip(Y_list, new_tau_list)], axis=0)
    return new_Y_avg, new_tau_list
    

if __name__ == '__main__':
    # Parameters
    n_sub = 20  # must be even to plot the sources
    p = 15
    n = 2000
    noise = 1

    # Generate data
    X_list, A_list, S, S_list, true_tau_list = _generate_data(n_sub, p, n)
    noise_unmixing = 0.2  # can be modified
    W_list = np.array([np.linalg.inv(a) + noise_unmixing * np.random.randn(p, p) for a in A_list])

    Y_list = np.array([np.dot(w, x) for w, x in zip(W_list, X_list)])
    Y_avg = np.mean(Y_list, axis=0)
    tau_list = np.zeros(n_sub, dtype=int)

    # Optimization
    loss = []
    n_iter = 8
    for i in range(n_iter):
        loss.append(_loss_function(W_list, X_list, Y_avg, noise, tau_list))
        Y_avg, tau_list = _optimization_tau_step(Y_list, Y_avg, tau_list)
    
    # print("True tau : {}".format(true_tau_list - true_tau_list[0]))
    # print("Estimated tau : {}".format(tau_list - tau_list[0]))

    # Plot
    plt.plot(loss)
    plt.title("Negative log-likelihood of the model (without function f)")
    plt.xlabel("Iterations")
    plt.ylabel("NLL")
    plt.savefig("figures/loss_optim_tau.pdf")
    plt.show()

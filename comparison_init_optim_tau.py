import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from joblib import Parallel, delayed


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
    S_list = np.array([np.roll(s, tau) for s, tau in zip(S_list, tau_list)])
    return S_list, tau_list


def _loss_function(Y_list, Y_avg):
    n_sub, p, _ = Y_list.shape
    loss = 0
    for Y in Y_list:
        Y_denoise = (n_sub * Y_avg - Y) / (n_sub - 1)
        loss += np.mean((Y - Y_denoise) ** 2) * p
    return loss


def new_correlate(X, Y):
    _, n = np.shape(X)
    corr = np.array([np.sum(X * np.roll(Y, delay, axis=1), axis=1)
                    for delay in np.arange(-n+1, n)]).T
    corr_norm = np.sum(corr, axis=0)
    new_tau = np.argmax(corr_norm) - n + 1
    return new_tau


def _optimization_tau_step(Y_list, Y_avg):
    n_sub, _, _ = Y_list.shape
    new_tau_list = np.zeros(n_sub, dtype=int)
    for i in range(n_sub):
        new_tau_list[i] = new_correlate(Y_list[i], Y_avg)
        Y_avg -= Y_list[i] / n_sub
        Y_list[i] = np.roll(Y_list[i], -new_tau_list[i], axis=1)
        Y_avg += Y_list[i] / n_sub
    return Y_list, Y_avg, new_tau_list


def run_experiment(n_sub, p, n, sources_noise, random_state, init):
    S_list, true_tau_list = _create_sources(n_sub, p, n, sources_noise, random_state)
    Y_list = np.copy(S_list)
    if init == 'mean':
        Y_avg = np.mean(S_list, axis=0)
    elif init == 'first_subject':
        Y_avg = S_list[0]
    else:
        print("Wrong initialization name \n")
    tau_list = np.zeros(n_sub, dtype=int)

    # Optimization
    loss = []
    n_iter = 8
    for _ in range(n_iter):
        loss.append(_loss_function(Y_list, Y_avg))
        Y_list, Y_avg, new_tau_list = _optimization_tau_step(Y_list, Y_avg)
        tau_list += new_tau_list
    loss = np.array(loss)

    delay_error = true_tau_list - true_tau_list[0] - tau_list[0] + tau_list
    output = {"Loss": loss, "Delay": delay_error, "True delay": true_tau_list}

    return output


if __name__ == '__main__':
    # Parameters
    n_sub = 4
    p = 3
    n = 200
    sources_noise = 1
    nb_expe = 10
    random_state = range(nb_expe)
    N_JOBS = 4

    # Compute average loss
    results_mean = Parallel(n_jobs=N_JOBS)(delayed(run_experiment)(
        n_sub, p, n, sources_noise, r, 'mean') for r in random_state)
    results_mean = pd.DataFrame(results_mean)
    loss_mean = np.mean(results_mean['Loss'], axis=0)
    delay_error_mean = np.mean(results_mean['Delay'], axis=0)

    results_first_sub = Parallel(n_jobs=N_JOBS)(delayed(run_experiment)(
        n_sub, p, n, sources_noise, r, 'first_subject') for r in random_state)
    results_first_sub = pd.DataFrame(results_first_sub)
    loss_first_sub = np.mean(results_first_sub['Loss'])
    delay_error_first_sub = np.mean(results_first_sub['Delay'], axis=0)

    # print("Delay error mean : {}".format(delay_error_mean))
    # print("Delay error first subject : {}".format(delay_error_first_sub))

    # print("Loss mean : {}".format(loss_mean))
    # print("Loss first subject : {}".format(loss_first_sub))

    # Plot
    plt.semilogy(loss_mean, label='Init. mean')
    plt.semilogy(loss_first_sub, label='Init. first subject')
    plt.title(
        "Negative log-likelihood of the model (without function f)", fontweight="bold")
    plt.xlabel("Iterations")
    plt.ylabel("NLL (logscale)")
    plt.legend()
    plt.grid()
    plt.savefig("figures/init_optim_tau_nsub" + str(n_sub) + "_p" +
                str(p) + "_n" + str(n) + "_noise" + str(sources_noise) + ".pdf")
    plt.show()

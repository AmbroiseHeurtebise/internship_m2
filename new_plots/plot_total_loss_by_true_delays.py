import numpy as np
import pandas as pd
from multiviewica import multiviewica
from delay_multiviewica import multiviewica_test5 as delay_multiviewica_test5
from delay_multiviewica import create_sources_pierre, univiewica, _apply_delay
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))


def _total_loss_function(W_list, Y_list, Y_avg, noise=1):
    _, p, _ = W_list.shape
    loss = np.mean(_logcosh(Y_avg)) * p
    for (W, Y) in zip(W_list, Y_list):
        loss -= np.linalg.slogdet(W)[1]
        loss += 1 / (2 * noise) * np.mean((Y - Y_avg) ** 2) * p
    return loss


def run_experiment(m, p, n, algo, delay_max, random_state):
    X_list, _, true_tau_list = create_sources_pierre(
        m, p, n, delay_max, sigma=0.05, random_state=random_state)
    if algo == 'mvica_true_delays':
        new_X_list = _apply_delay(X_list, -true_tau_list)
        _, W_list, _ = multiviewica(new_X_list, random_state=random_state)
        S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
        Y_list = _apply_delay(S_list, -true_tau_list)
        Y_avg = np.mean(Y_list, axis=0)
        total_loss = _total_loss_function(W_list, Y_list, Y_avg)
    elif algo == 'delay_mvica_true_delays':
        _, W_list, _, tau_list = delay_multiviewica_test5(
            X_list, tau_list=-true_tau_list, random_state=random_state)
        S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
        Y_list = _apply_delay(S_list, tau_list)
        Y_avg = np.mean(Y_list, axis=0)
        total_loss = _total_loss_function(W_list, Y_list, Y_avg)
    else:
        new_X_list = _apply_delay(X_list, -true_tau_list)
        W_list = univiewica(new_X_list, random_state=random_state)
        W_list = np.array(W_list)
        S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
        Y_list = _apply_delay(S_list, -true_tau_list)
        Y_avg = np.mean(Y_list, axis=0)
        total_loss = _total_loss_function(W_list, Y_list, Y_avg)
    output = {"Algo": algo, "Delay": delay_max,
              "random_state": random_state, "Total loss": total_loss}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 2
    n = 400
    delays = np.linspace(0, n // 3, 20, dtype=int)
    algos = ['mvica_true_delays', 'delay_mvica_true_delays', 'univiewICA_true_delays']
    n_expe = 20
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(m, p, n, algo, delay_max, random_state)
        for algo, delay_max, random_state
        in product(algos, delays, range(n_expe))
    )
    results = pd.DataFrame(results).drop(columns='random_state')

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Delay",
                       y="Total loss", hue="Algo", linewidth=2.5)
    # fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Loss")
    leg = plt.legend(prop={'size': 10})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Total loss wrt the true delays", fontsize=18, fontweight="bold")
    plt.savefig("new_figures/total_loss_by_true_delays.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

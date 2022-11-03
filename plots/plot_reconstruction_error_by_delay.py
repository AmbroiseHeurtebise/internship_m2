import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
from multiviewica_delay import multiviewica, multiviewica_delay, univiewica, _apply_delay_one_sub, generate_data


# def reconstruction_error(Y_avg, S):
#     assert Y_avg.shape == S.shape
#     p, _ = Y_avg.shape
#     tau = _delay_estimation(Y_avg, S)
#     Y_avg = _apply_delay_one_sub(Y_avg, tau)
#     M = np.dot(Y_avg, S.T)
#     error = amari_distance(M, np.eye(p))
#     return error


def reconstruction_error(Y_avg, S):
    assert Y_avg.shape == S.shape
    p, n = Y_avg.shape
    Y_avg_list = np.array(
        [_apply_delay_one_sub(Y_avg, tau) for tau in range(n)])
    M_list = np.array([np.dot(Y, S.T) for Y in Y_avg_list])
    error_list = [amari_distance(M, np.eye(p)) for M in M_list]
    error = np.min(error_list)
    return error


mem = Memory(".")


@mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, algo, delay_max, noise, random_state
):
    # X_list, _, _, _, S = create_model(
    #     m, p, n, delay_max, noise, random_state=random_state
    # )
    # X_list, _, _, S = create_sources_pierre(
    #     m, p, n, delay_max, noise, random_state)
    X_list, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise, random_state=random_state)
    if algo == 'MVICA':
        _, _, Y_avg = multiviewica(X_list, random_state=random_state)
    elif algo == 'MVICAD':
        _, _, Y_avg, _ = multiviewica_delay(
            X_list, optim_delays_ica=False, random_state=random_state)
    elif algo == 'UniviewICA':
        W_list = univiewica(X_list, random_state=random_state)
        Y_list = np.array([np.dot(W, X) for W, X in zip(W_list, X_list)])
        Y_avg = np.mean(Y_list, axis=0)
    else:
        raise ValueError("Wrong algo name")
    rec_error = reconstruction_error(Y_avg, S)
    output = {"Algo": algo, "Delay": delay_max, "random_state": random_state,
              "Reconstruction_error": rec_error}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 10
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    algos = ['MVICA', 'MVICAD', 'UniviewICA']
    delays = np.linspace(0, n * 0.5, 11, dtype=int)
    noise = 0.05
    n_expe = 25
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, algo, delay_max, noise,
            random_state)
        for algo, delay_max, random_state
        in product(algos, delays, range(n_expe))
    )
    results = pd.DataFrame(results)

    # Lineplots
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Delay",
                       y="Reconstruction_error", hue="Algo", linewidth=2.5)
    fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Reconstruction error")
    plt.grid()
    plt.title(
        "Reconstruction error wrt the delay", fontsize=18, fontweight="bold")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.savefig("figures/reconstruction_error_by_delay.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

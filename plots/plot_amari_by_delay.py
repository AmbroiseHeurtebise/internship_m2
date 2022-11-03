import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
from multiviewica_delay import multiviewica, multiviewica_delay, univiewica, generate_data


mem = Memory(".")


@mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, algo, delay_max, noise, random_state
):
    X_list, A_list, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise, random_state=random_state)
    if algo == 'MVICA':
        _, W_list, _ = multiviewica(X_list, random_state=random_state)
    elif algo == 'mvica_with_optim_permica':
        _, W_list, _ = multiviewica(
            X_list, optim_delays_permica=True, random_state=random_state)
    elif algo == 'delay_mvica_without_optim_permica':
        _, W_list, _, _ = multiviewica_delay(
            X_list, optim_delays_permica=False, random_state=random_state)
    elif algo == 'MVICAD':
        _, W_list, _, _ = multiviewica_delay(
            X_list, optim_delays_ica=False, random_state=random_state)
    elif algo == 'delay_mvica_without_optim_permica_and_ica':
        _, W_list, _, _ = multiviewica_delay(
            X_list, optim_delays_permica=False, optim_delays_ica=False,
            random_state=random_state)
    elif algo == 'mvicad':
        _, W_list, _, _ = multiviewica_delay(X_list, random_state=random_state)
    elif algo == 'UniviewICA':
        W_list = univiewica(X_list, random_state=random_state)
    else:
        raise ValueError("Wrong algo name")
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Delay": delay_max,
              "random_state": random_state, "Amari_distance": amari}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 10
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    # algos = [
    #     'mvica', 'delay_mvica_without_optim_permica',
    #     'delay_mvica_without_optim_ica', 'delay_mvica']
    # algos = ['mvica', 'univiewica', 'delay_mvica']
    algos = ['MVICA', 'MVICAD', 'UniviewICA']
    delays = np.linspace(0, n * 0.5, 10, dtype=int)
    noise = 0.5
    n_expe = 2
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, algo, delay_max, noise,
            random_state)
        for algo, delay_max, random_state
        in product(algos, delays, range(n_expe))
    )
    results = pd.DataFrame(results).drop(columns='random_state')

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Delay",
                       y="Amari_distance", hue="Algo", linewidth=2.5)
    fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Amari distance")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    # for i, l in enumerate(leg.get_texts()):
    #     if i == 1:
    #         l.set_weight('bold')
    plt.grid()
    plt.title("Amari distance wrt delay", fontsize=18, fontweight="bold")
    plt.savefig("figures/amari_by_delay.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

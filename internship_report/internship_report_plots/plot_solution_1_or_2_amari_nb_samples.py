import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
from multiviewica_delay import multiviewica_delay, generate_data


mem = Memory(".")


@mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, treshold, algo, delay_max, noise, random_state
):
    X_list, A_list, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=noise, random_state=random_state)
    if algo == 'Algorithm 4 alone':
        _, W_list, _, _, _ = multiviewica_delay(
            X_list, optim_delays_ica=False, random_state=random_state)
    elif algo == 'Algorithms 3 and 4 together':
        _, W_list, _, _, _ = multiviewica_delay(X_list, random_state=random_state)
    else:
        raise ValueError("Wrong algo name")
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Number of samples": n,
              "random_state": random_state, "Amari_distance": amari}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 10
    nb_samples = np.linspace(50, 700, 14, dtype=int)
    nb_intervals = 5
    nb_freqs = 20
    treshold = 1
    algos = ['Algorithm 4 alone', 'Algorithms 3 and 4 together']
    delay_max = 10
    noise = 1
    n_expe = 2
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, treshold, algo, delay_max, noise,
            random_state)
        for n, algo, random_state
        in product(nb_samples, algos, range(n_expe))
    )
    results = pd.DataFrame(results).drop(columns='random_state')

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Number of samples",
                       y="Amari_distance", hue="Algo", linewidth=2.5)
    fig.set(yscale='log')
    x_ = plt.xlabel("Number of samples")
    y_ = plt.ylabel("Amari distance")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title(
        "Amari distance wrt number of samples", fontsize=18,
        fontweight="bold")
    plt.savefig(
        "internship_report_figures/solution_1_or_2_amari_nb_samples.pdf",
        bbox_extra_artists=[x_, y_], bbox_inches="tight")

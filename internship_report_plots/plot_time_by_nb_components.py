import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from itertools import product
from joblib import Parallel, delayed, Memory
from multiviewica_delay import multiviewica, multiviewica_delay, generate_data


# mem = Memory(".")


# @mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, algo, delay_max, noise, random_state
):
    X_list, _, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise, random_state=random_state)
    start_time = time.time()
    if algo == 'MVICA':
        multiviewica(X_list, random_state=random_state)
    elif algo == 'MVICAD':
        multiviewica_delay(
            X_list, optim_delays_ica=False, random_state=random_state)
    else:
        raise ValueError("Wrong algo name")
    total_time = time.time() - start_time
    output = {"Algo": algo, "Number of components": p,
              "random_state": random_state, "Time": total_time}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    nb_components = np.arange(1, 20, 2)
    n = 50
    nb_intervals = 5
    nb_freqs = 20
    algos = ['MVICA', 'MVICAD']
    delay_max = 50
    noise = 0.5
    n_expe = 5
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, algo, delay_max, noise,
            random_state)
        for p, algo, random_state
        in product(nb_components, algos, range(n_expe))
    )
    results = pd.DataFrame(results).drop(columns='random_state')

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Number of components",
                       y="Time", hue="Algo", linewidth=2.5)
    fig.set(yscale='log')
    x_ = plt.xlabel("Number of components")
    y_ = plt.ylabel("Time execution (s)")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title(
        "Time execution wrt number of components", fontsize=18,
        fontweight="bold")
    plt.savefig(
        "internship_report_figures/time_by_nb_components.pdf",
        bbox_extra_artists=[x_, y_], bbox_inches="tight")

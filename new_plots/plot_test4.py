# Fourth test: vary the parameter n_iter_delay

import numpy as np
import pandas as pd
from multiviewica import multiviewica
from delay_multiviewica import multiviewica as delay_multiviewica
from delay_multiviewica import create_sources_pierre, univiewica
from picard import amari_distance
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed


def run_experiment(m, p, n, algo, delay_max, random_state):
    X_list, A_list, _ = create_sources_pierre(
        m, p, n, delay_max, sigma=0.05, random_state=random_state)
    if algo == 'mvica':
        _, W_list, _ = multiviewica(X_list, random_state=random_state)
    elif algo == 'delay_mvica':
        _, W_list, _, _ = delay_multiviewica(X_list, random_state=random_state)
    elif algo == 'delay_mvica_1iter':
        _, W_list, _, _ = delay_multiviewica(
            X_list, n_iter_delay=1, random_state=random_state)
    elif algo == 'delay_mvica_3iter':
        _, W_list, _, _ = delay_multiviewica(
            X_list, n_iter_delay=3, random_state=random_state)
    elif algo == 'delay_mvica_4iter':
        _, W_list, _, _ = delay_multiviewica(
            X_list, n_iter_delay=4, random_state=random_state)
    else:
        W_list = univiewica(X_list, random_state=random_state)
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Delay": delay_max,
              "random_state": random_state, "Amari_distance": amari}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 2
    n = 400
    delays = np.linspace(0, n // 1.5, 6, dtype=int)
    algos = ['delay_mvica', 'delay_mvica_1iter',
             'delay_mvica_3iter', 'delay_mvica_4iter']
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
                       y="Amari_distance", hue="Algo", linewidth=2.5)
    fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Amari distance")
    leg = plt.legend(prop={'size': 10})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Amari distance wrt the delay", fontsize=18, fontweight="bold")
    plt.savefig("new_figures/test4.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

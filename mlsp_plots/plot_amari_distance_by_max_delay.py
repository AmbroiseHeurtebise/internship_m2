import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
from multiviewica_delay import (
    multiviewica,
    multiviewica_delay,
    generate_data
)


mem = Memory(".")


@mem.cache
def run_experiment(
    m,
    p,
    n,
    nb_intervals,
    nb_freqs,
    treshold,
    algo,
    max_delay,
    noise,
    random_state
):
    X_list, A_list, _, _, _ = generate_data(
        m=m, p=p, n=n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=noise,
        random_state=random_state, shared_delays=False)
    if algo == 'MVICA':
        _, W_list, _ = multiviewica(
            X_list,
            random_state=random_state
        )
    elif algo == 'MVICAD':
        _, W_list, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=max_delay,
            optim_delays_permica=False,  # XXX just a test
            every_n_iter_delay=10,
            random_state=random_state,
            shared_delays=True,
        )
    elif algo == 'MVICAD_mul_delays':
        _, W_list, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=max_delay,
            every_n_iter_delay=10,
            random_state=random_state,
            shared_delays=False,
        )
    else:
        raise ValueError("Wrong algo name")
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Delay": max_delay,
              "random_state": random_state, "Amari_distance": amari}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 10
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    treshold = 1
    algos = ['MVICA', 'MVICAD', 'MVICAD_mul_delays']
    delays = np.linspace(0, 20, 11, dtype=int)
    noise = 1
    n_expe = 5
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, treshold, algo, max_delay, noise,
            random_state)
        for algo, max_delay, random_state
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
    plt.savefig("mlsp_figures/amari_by_delay_multiple_%s_seeds.jpeg" % n_expe,
                bbox_extra_artists=[x_, y_], bbox_inches="tight")

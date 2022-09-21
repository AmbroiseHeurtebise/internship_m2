import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
from multiviewica import multiviewica
from delay_multiviewica import delay_multiviewica, create_sources_pierre, univiewica


mem = Memory(".")


@mem.cache
def run_experiment(m, p, n, algo, delay_max, random_state):
    X_list, A_list, _, _ = create_sources_pierre(
        m, p, n, delay_max, sigma=0.05, random_state=random_state)
    if algo == 'mvica':
        _, W_list, _ = multiviewica(X_list, random_state=random_state)
    elif algo == 'delay_mvica':
        _, W_list, _, _ = delay_multiviewica(X_list, random_state=random_state)
    elif algo == 'univiewica':
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
    p = 2
    n = 400
    delays = np.linspace(0, n * 0.65, 9, dtype=int)
    algos = ['mvica', 'univiewica', 'delay_mvica']
    n_expe = 17
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(m, p, n, algo, delay_max, random_state)
        for algo, delay_max, random_state
        in product(algos, delays, 4 + np.arange(n_expe))
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
    plt.savefig("new_figures/amari_by_delay.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

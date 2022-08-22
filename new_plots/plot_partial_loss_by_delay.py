import numpy as np
import pandas as pd
from multiviewica import multiviewica
from delay_multiviewica import multiviewica as delay_multiviewica
from delay_multiviewica import create_sources_pierre, univiewica, _loss_delay
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed


def run_experiment(m, p, n, algo, delay_max, random_state):
    X_list, _, _ = create_sources_pierre(
        m, p, n, delay_max, sigma=0.05, random_state=random_state)
    if algo == 'mvica':
        _, W_list, _ = multiviewica(X_list, random_state=random_state)
        S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
        partial_loss = _loss_delay(S_list, tau_list=np.zeros(m, dtype=int))
    elif algo == 'delay_multiviewica':
        _, W_list, _, tau_list = delay_multiviewica(X_list, random_state=random_state)
        S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
        partial_loss = _loss_delay(S_list, tau_list)
    else:
        W_list = univiewica(X_list, random_state=random_state)
        W_list = np.array(W_list)
        S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
        partial_loss = _loss_delay(S_list, tau_list=np.zeros(m, dtype=int))
    output = {"Algo": algo, "Delay": delay_max,
              "random_state": random_state, "Partial loss": partial_loss}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 2
    n = 400
    delays = np.linspace(0, n // 3, 20, dtype=int)
    algos = ['univiewICA', 'mvica', 'delay_multiviewica']
    n_expe = 5
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
                       y="Partial loss", hue="Algo", linewidth=2.5)
    # fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Loss")
    leg = plt.legend(prop={'size': 10})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Partial loss wrt the delay", fontsize=18, fontweight="bold")
    plt.savefig("new_figures/partial_loss_by_delay.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

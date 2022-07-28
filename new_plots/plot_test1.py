import numpy as np
import pandas as pd
from multiviewica import multiviewica
from delay_multiviewica import multiviewica as delay_multiviewica
from delay_multiviewica import multiviewica_test1 as delay_multiviewica_test1
from picard import picard, amari_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_random_state
from itertools import product
from joblib import Parallel, delayed


def delay_source(x, tau):
    return np.roll(x, -tau, axis=1)


def create_sources_pierre(n, m, delay_max, sigma=0.05, random_state=None):
    rng = check_random_state(random_state)
    delays = rng.randint(0, 1 + delay_max, m)
    s1 = np.zeros(n)
    s1[:n//2] = np.sin(np.linspace(0, np.pi, n//2))
    s2 = rng.randn(n) / 10
    S = np.c_[s1, s2].T
    N = sigma * rng.randn(m, p, n)
    A_list = rng.randn(m, p, p)
    X_list = np.array([np.dot(A, delay_source(S, delay) + noise)
                      for A, noise, delay in zip(A_list, N, delays)])
    return X_list, A_list


def univiewica(X_list, random_state=None):
    W_list = []
    for x in X_list:
        K, W, _ = picard(x, random_state=random_state)
        W_list.append(np.dot(W, K))
    return W_list


def run_experiment(n, m, algo, delay_max, random_state, stop_optim_delay=100):
    X_list, A_list = create_sources_pierre(
        n, m, delay_max, sigma=0.05, random_state=random_state)
    if algo == 'mvica':
        _, W_list, _ = multiviewica(X_list, random_state=random_state)
    elif algo == 'delay_mvica':
        _, W_list, _, _ = delay_multiviewica(X_list, random_state=random_state)
    elif algo == 'delay_mvica_stopAPCR':
        _, W_list, _, _ = delay_multiviewica_test1(X_list, stop_optim_delay=stop_optim_delay, random_state=random_state)
    else:
        W_list = univiewica(X_list, random_state=random_state)
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Delay": delay_max,
              "random_state": random_state, "Amari_distance": amari}
    return output


if __name__ == '__main__':
    # Parameters
    n = 400
    m = 6
    p = 2
    delays = np.linspace(0, n // 1.5, 6, dtype=int)
    algos = ['univiewICA', 'mvica', 'delay_mvica', 'delay_mvica_stopAPCR']
    n_expe = 10
    N_JOBS = 4
    stop_optim_delay = 100

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(n, m, algo, delay_max, random_state, stop_optim_delay)
        for algo, delay_max, random_state
        in product(algos, delays, range(n_expe))
    )
    results = pd.DataFrame(results).drop(columns='random_state')
    results = results.groupby(['Algo', 'Delay']).mean()

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
    plt.savefig("figures/amari_delaymvica_mvica.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

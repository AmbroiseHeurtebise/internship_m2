import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed
from picard import amari_distance
from multiviewica import multiviewica
from delay_multiviewica import delay_multiviewica, create_sources_pierre, univiewica


def reconstruction_error(S1, S2):
    assert S1.shape == S2.shape
    _, p, _ = S1.shape
    M_list = np.array([np.dot(s1, s2.T) for s1, s2 in zip(S1, S2)])
    error = np.sum([amari_distance(M, np.eye(p)) for M in M_list])
    return error


def run_experiment(m, p, n, algo, delay_max, random_state):
    X_list, A_list, _ = create_sources_pierre(
        m, p, n, delay_max, sigma=0.05, random_state=random_state)
    true_W_list = np.array([np.linalg.inv(A) for A in A_list])
    true_S_list = np.array([np.dot(W, X) for W, X in zip(true_W_list, X_list)])
    if algo == 'mvica':
        _, W_list, _ = multiviewica(X_list, random_state=random_state)
    elif algo == 'delay_mvica':
        _, W_list, _, _ = delay_multiviewica(X_list, random_state=random_state)
    elif algo == 'univiewICA':
        W_list = univiewica(X_list, random_state=random_state)
    else:
        raise ValueError("Wrong algo name")
    S_list = np.array([np.dot(W, X) for W, X in zip(W_list, X_list)])
    rec_error = reconstruction_error(S_list, true_S_list)
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Delay": delay_max,
              "random_state": random_state, "Amari_distance": amari, 
              "Reconstruction_error": rec_error}
    return output


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 2
    n = 400
    delays = np.linspace(0, n * 0.65, 9, dtype=int)
    algos = ['mvica', 'delay_mvica']  # 'univiewICA'
    n_expe = 20
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(m, p, n, algo, delay_max, random_state)
        for algo, delay_max, random_state
        in product(algos, delays, range(n_expe))
    )
    results = pd.DataFrame(results)  # .drop(columns='random_state')

    # Lineplots
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Delay",
                       y="Amari_distance", hue="Algo", linewidth=2.5)
    fig = sns.lineplot(data=results, x="Delay",
                       y="Reconstruction_error", hue="Algo", linewidth=2.5, 
                       linestyle='--')
    fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Amari distance and reconstruction_error")
    plt.grid()
    plt.title("Amari distance wrt the delay", fontsize=18, fontweight="bold")
    
    # Legend
    leg = plt.legend(prop={'size': 10})
    leg.get_texts()[0].set_text('amari_mvica')
    leg.get_texts()[1].set_text('amari_delay_mvica')
    leg.get_texts()[2].set_text('error_rec_mvica')
    leg.get_texts()[3].set_text('error_rec_delay_mvica')
    nb_lines = 2 * len(algos)
    for i, line in zip(range(nb_lines), leg.get_lines()):
        line.set_linewidth(2.5)
        line.set_label('bla')
        if i >= nb_lines // 2:
            line.set_linestyle('--')
    
    plt.savefig("new_figures/reconstruction_error_by_delay.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

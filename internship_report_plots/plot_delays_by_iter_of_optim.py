import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from multiviewica_delay import (
    _optimization_tau,
    _optimization_tau_approach1,
    _optimization_tau_approach2,
    generate_data
)

mem = Memory(".")


@mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, treshold, delay_max, noise, n_iter,
    random_state, init
):
    # Generate data
    _, _, true_tau_list, S_list, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=noise,
        random_state=random_state)

    # Optimization
    if init == 'Approach 1':
        _, _, _, delay_estim_error = _optimization_tau_approach1(
            S_list, n_iter, error_tau=True, true_tau_list=true_tau_list)
    elif init == 'Approach 2':
        _, _, _, delay_estim_error = _optimization_tau_approach2(
            S_list, n_iter, error_tau=True, true_tau_list=true_tau_list)
    elif init == 'Combined approach':
        _, _, _, delay_estim_error = _optimization_tau(
            S_list, n_iter, error_tau=True, true_tau_list=true_tau_list)
    else:
        raise ValueError(
            "init should be approach_1, approach_2 or combined_approach")
    delay_estim_error = np.array(delay_estim_error)

    output = {"Init": init, "random_state": random_state,
              "Delay estimation error": delay_estim_error}
    return output


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 6
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    treshold = 1
    delay_max = 10
    noise = 1
    init = ['Approach 1', 'Approach 2', 'Combined approach']
    n_iter = 2
    nb_expe = 10
    random_states = range(nb_expe)
    N_JOBS = 8

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, treshold, delay_max, noise,
            n_iter, r, i)
        for r, i
        in product(random_states, init)
    )
    results = pd.DataFrame(results)
    delay_estim_error = pd.DataFrame(
        data=[x for x in results['Delay estimation error'].values])
    results = pd.concat(
        [results.drop(columns=['Delay estimation error']), delay_estim_error],
        axis=1)
    results = results.melt(
        id_vars=["Init", "random_state"],
        var_name="Iteration",
        value_name="Delay estimation error")

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Iteration",
                       y="Delay estimation error", hue="Init", linewidth=2.5)
    x_ = plt.xlabel("Iterations (of the outer loop)")
    y_ = plt.ylabel("Delay estimation error")
    plt.xticks(np.arange(n_iter + 1) * m, np.arange(n_iter + 1))
    leg = plt.legend(prop={'size': 10})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Distance between estimated delays and true delays",
              fontsize=18, fontweight="bold")
    plt.savefig("internship_report_figures/delays_by_iter_of_optim.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from multiviewica_delay import generate_data, multiviewica_delay


def distance_between_delays(tau1, tau2, n):
    assert len(tau1) == len(tau2)
    n_sub = len(tau1)
    error = 0
    for j in range(n_sub):
        error += np.min(np.abs(
            [tau1[j] - tau2[j] - n, tau1[j] - tau2[j], tau1[j] - tau2[j] + n]))
    # error /= n_sub
    return error


mem = Memory(".")


@mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, delay_max, noise, random_state
):
    # Generate data
    X_list, _, true_tau_list, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise,
        random_state=random_state)

    # Estimate delays
    _, _, _, tau_list, tau_list_init = multiviewica_delay(
        X_list, n_iter_delay=1, random_state=random_state)

    # Errors
    error_init = distance_between_delays(
        tau_list_init, true_tau_list - true_tau_list[0], n)
    error_final = distance_between_delays(
        tau_list, true_tau_list - true_tau_list[0], n)

    # Output
    output = {"Noise": noise, "Random state": random_state,
              "Error init": error_init, "Error final": error_final}
    return output


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 10
    noise_list = np.logspace(-0.6, 0.5, 12)
    # noise_list = np.logspace(-0.6, 1, 17)
    nb_expe = 25
    random_states = np.arange(nb_expe)
    N_JOBS = 4

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, delay_max, noise, random_state)
        for noise, random_state
        in product(noise_list, random_states)
    )
    results = pd.DataFrame(results)

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error init", linewidth=2.5, label="Phase 1")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error final", linewidth=2.5, label="Phases 1 and 2")
    fig.set(xscale='log')
    fig.set(yscale='log')
    x_ = plt.xlabel("Noise")
    y_ = plt.ylabel("Error")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Delay estimation error for various noise levels")
    plt.savefig("tests_figures/importance_optim_phase_2.pdf")
    plt.show()

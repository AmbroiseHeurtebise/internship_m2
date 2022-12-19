import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_random_state
from itertools import product
from joblib import Parallel, delayed, Memory
from multiviewica_delay import generate_data, multiviewica_delay


def normalize_delays(tau_list, n):
    tau_list -= tau_list[0]
    tau_list %= n
    return tau_list


def estimation_signal_power(
    m, p, n, nb_intervals, nb_freqs, delay_max, n_seeds=100, rng=None
):
    rng = check_random_state(rng)
    power = []
    seeds = rng.randint(0, 10000, n_seeds)
    for random_state in seeds:
        _, _, _, _, S = generate_data(
            m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
            treshold=3, delay=delay_max, noise=0.1,
            random_state=random_state)
        power.append(np.mean(S ** 2))
    return np.mean(power)


def distance_between_delays(tau1, tau2, n):
    assert len(tau1) == len(tau2)
    n_sub = len(tau1)
    error = 0
    for j in range(n_sub):
        error += np.min(np.abs(
            [tau1[j] - tau2[j] - n, tau1[j] - tau2[j], tau1[j] - tau2[j] + n]))
    return error


mem = Memory(".")


@mem.cache
def run_experiment(
    m, p, n, nb_intervals, nb_freqs, delay_max, noise, random_state,
    signal_power=10
):
    # Generate data
    X_list, _, true_tau_list, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise,
        random_state=random_state)
    true_tau_list = normalize_delays(true_tau_list, n)

    # SNR
    if noise == 0:
        snr = 0
    else:
        snr = signal_power / (noise ** 2)

    # Estimate delays
    _, _, _, tau_list, tau_list_init = multiviewica_delay(
        X_list, delay_max=None, n_iter_delay=2, optim_approach=1, random_state=random_state)
    tau_list = normalize_delays(tau_list, n)
    tau_list_init = normalize_delays(tau_list_init, n)

    _, _, _, tau_list_with_f, _ = multiviewica_delay(
        X_list, delay_max=None, n_iter_f=2, optim_delays_with_f=True, use_f=True,
        random_state=random_state)
    tau_list_with_f = normalize_delays(tau_list_with_f, n)

    # Errors
    error_init = distance_between_delays(
        tau_list_init, true_tau_list, n)
    error_final = distance_between_delays(
        tau_list, true_tau_list, n)
    error_with_f = distance_between_delays(
        tau_list_with_f, true_tau_list, n)

    # Output
    output = {"Noise": noise, "SNR": snr, "Random state": random_state,
              "Error init": error_init, "Error final": error_final,
              "Error with f": error_with_f}
    return output


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 10
    noise_list = np.logspace(-0.6, 1, 17)
    nb_expe = 10
    random_states = np.arange(nb_expe)
    N_JOBS = 4

    # Estimate signal power
    signal_power = estimation_signal_power(
        m, p, n, nb_intervals, nb_freqs, delay_max, n_seeds=1000, rng=0)

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, delay_max, noise, random_state,
            signal_power)
        for noise, random_state
        in product(noise_list, random_states)
    )
    results = pd.DataFrame(results)

    # Save in csv file
    results.to_csv("results_importance_phase_2.csv")

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error init", linewidth=2.5, label="Phase 1")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error final", linewidth=2.5, label="Phases 1 and 2")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error with f", linewidth=2.5, label="Phases 1 and 2 with f")
    fig.set(xscale='log')
    fig.set(yscale='log')
    x_ = plt.xlabel("Noise")
    y_ = plt.ylabel("Error")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Delay estimation error for various noise levels")
    plt.savefig("tests_figures/importance_optim_phase_2.pdf", bbox_inches="tight")

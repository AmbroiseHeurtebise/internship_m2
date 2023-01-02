import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
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
    m, p, n, nb_intervals, nb_freqs, treshold, delay_max, noise, random_state,
    signal_power=10
):
    # Generate data
    X_list, A_list, true_tau_list, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=noise,
        random_state=random_state)
    true_tau_list = normalize_delays(true_tau_list, n)

    # SNR
    if noise == 0:
        snr = 0
    else:
        snr = signal_power / (noise ** 2)

    # Estimate delays
    W_list_init, tau_list_init, W_list, tau_list = multiviewica_delay(
        X_list, delay_max=delay_max, n_iter_delay=2, optim_approach=1,
        random_state=random_state, return_unmixing_delays_both_phases=True)
    tau_list += tau_list_init
    tau_list = normalize_delays(tau_list, n)
    tau_list_init = normalize_delays(tau_list_init, n)

    _, W_list_with_f, _, tau_list_with_f, _ = multiviewica_delay(
        X_list, delay_max=delay_max, n_iter_f=2, optim_delays_with_f=True,
        use_f=True, random_state=random_state)
    tau_list_with_f = normalize_delays(tau_list_with_f, n)

    # Errors
    error_init = distance_between_delays(
        tau_list_init, true_tau_list, n)
    error_final = distance_between_delays(
        tau_list, true_tau_list, n)
    error_with_f = distance_between_delays(
        tau_list_with_f, true_tau_list, n)

    # Amari distance
    amari_init = np.sum([amari_distance(W, A) for W, A in zip(W_list_init, A_list)])
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    amari_with_f = np.sum([amari_distance(W, A) for W, A in zip(W_list_with_f, A_list)])

    # Output
    output = {"Noise": noise, "SNR": snr, "Random state": random_state,
              "Error init": error_init, "Error final": error_final,
              "Error with f": error_with_f, "Amari init": amari_init,
              "Amari": amari, "Amari with f": amari_with_f}
    return output


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    treshold = 0.5
    delay_max = 10
    # noise_list = np.logspace(-0.6, 1, 17)
    noise_list = np.logspace(-0.6, 1.4, 11)
    nb_expe = 5
    random_states = np.arange(nb_expe)
    N_JOBS = 4

    # Estimate signal power
    signal_power = estimation_signal_power(
        m, p, n, nb_intervals, nb_freqs, delay_max, n_seeds=1000, rng=0)

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, treshold, delay_max, noise,
            random_state, signal_power)
        for noise, random_state
        in product(noise_list, random_states)
    )
    results = pd.DataFrame(results)

    # Save in csv file
    file_name = "m" + str(m) + "_p" + str(p) + "_nbintervals" + str(nb_intervals) + "_nbfreqs" + str(nb_freqs) + "_delaymax" + str(delay_max)
    results.to_csv("data/" + file_name + ".csv")

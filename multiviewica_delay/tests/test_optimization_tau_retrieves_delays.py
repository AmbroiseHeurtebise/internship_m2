import numpy as np
from multiviewica_delay import (
    generate_data,
    _optimization_tau,
    _optimization_tau_approach1,
    _optimization_tau_approach2,
    _optimization_tau_with_f
)


def normalize_delays(tau_list, n):
    tau_list -= tau_list[0]
    tau_list %= n
    return tau_list


def test_optimization_tau_retrieves_delays():
    random_state = np.random.randint(1000)
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    treshold = 0.5
    delay_max = 10
    snr = 10  # Signal to noise ratio

    # Generate data
    _, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=0.,
        random_state=random_state)
    signal_power = np.mean(S ** 2)
    square_noise = signal_power / snr

    # Re generate data with noise
    _, _, true_tau_list, S_list, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=square_noise,
        random_state=random_state)

    # Estimate delays with optimization_tau
    _, tau_list, _ = _optimization_tau(
        S_list, n_iter=2, delay_max=delay_max)
    _, tau_list_1, _ = _optimization_tau_approach1(
        S_list, n_iter=2, delay_max=delay_max)
    _, tau_list_2, _ = _optimization_tau_approach2(
        S_list, n_iter=2, delay_max=delay_max)
    tau_list_with_f = _optimization_tau_with_f(
        S_list, n_iter=2, delay_max=delay_max)

    # Normalize delays
    true_tau_list = normalize_delays(true_tau_list, n)
    tau_list = normalize_delays(tau_list, n)
    tau_list_1 = normalize_delays(tau_list_1, n)
    tau_list_2 = normalize_delays(tau_list_2, n)
    tau_list_with_f = normalize_delays(tau_list_with_f, n)

    # Test if arrays are equal
    np.testing.assert_array_equal(tau_list, true_tau_list)
    np.testing.assert_array_equal(tau_list_1, true_tau_list)
    np.testing.assert_array_equal(tau_list_2, true_tau_list)
    np.testing.assert_array_equal(tau_list_with_f, true_tau_list)

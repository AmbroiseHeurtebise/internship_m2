import numpy as np
import pytest
from multiviewica_delay.multiviewica_shifts._apply_shifts import (
    _optimization_tau,
    _optimization_tau_approach1,
    _optimization_tau_approach2,
    _optimization_tau_with_f
)
from multiviewica_delay.multiviewica_shifts._generate_data import generate_data


def normalize_delays(tau_list, n):
    tau_list -= tau_list[0]
    tau_list %= n
    return tau_list


@pytest.mark.parametrize(
    "mode", ["base", "approach_1", "approach_2", "with_f"])
def test_optimization_tau_retrieves_delays(mode):
    random_state = 42
    m = 2
    p = 3
    n = 20
    nb_intervals = 2
    nb_freqs = 5
    treshold = 0.5
    max_delay = 10
    snr = 10  # Signal to noise ratio

    # Generate data
    _, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=0.,
        random_state=random_state)
    signal_power = np.mean(S ** 2)
    square_noise = signal_power / snr

    # Re generate data with noise
    _, _, true_tau_list, S_list, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=square_noise,
        random_state=random_state)

    # Estimate delays with optimization_tau
    if mode == "base":
        tau_list = _optimization_tau(
            S_list, n_iter=2, max_delay=max_delay)
    elif mode == "approach_1":
        tau_list = _optimization_tau_approach1(
            S_list, n_iter=2, max_delay=max_delay)
    elif mode == "approach_2":
        tau_list = _optimization_tau_approach2(
            S_list, n_iter=2, max_delay=max_delay)
    elif mode == "with_f":
        tau_list = _optimization_tau_with_f(
            S_list, n_iter=2, max_delay=max_delay)

    # Normalize delays
    true_tau_list = normalize_delays(true_tau_list, n)
    tau_list = normalize_delays(tau_list, n)

    # Test if arrays are equal
    np.testing.assert_array_equal(tau_list, true_tau_list)

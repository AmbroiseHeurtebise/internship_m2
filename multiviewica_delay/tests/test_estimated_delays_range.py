import numpy as np
import pytest
from multiviewica_delay.multiviewica_shifts._generate_data import generate_data
from multiviewica_delay import mvica_s


@pytest.mark.parametrize(
    "mode", ["base", "approach_1", "approach_2", "with_f"])
def test_estimated_delays_range(mode):
    random_state = 42
    m = 2
    p = 3
    n = 20
    nb_intervals = 2
    nb_freqs = 5
    treshold = 0.5
    max_delay = 10
    snr = 5  # Signal to noise ratio

    # Generate data
    _, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=0.,
        random_state=random_state)
    signal_power = np.mean(S ** 2)
    square_noise = signal_power / snr

    # Re generate data with noise
    X_list, _, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=square_noise,
        random_state=random_state)

    if mode == "base":
        kwargs = dict()
    elif mode == "approach_1":
        kwargs = dict(optim_approach=1)
    elif mode == "approach_2":
        kwargs = dict(optim_approach=2)
    elif mode == "with_f":
        kwargs = dict(optim_delays_with_f=True)

    # Estimate delays with MVICAD
    _, _, _, _, tau_list, _ = mvica_s(
        X_list, max_delay=max_delay, n_iter_delay=2, random_state=random_state,
        **kwargs)

    outsiders = np.arange(max_delay+1, n-max_delay)
    assert np.sum(np.isin(tau_list, outsiders)) == 0

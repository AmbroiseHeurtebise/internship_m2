import numpy as np
from multiviewica_delay import generate_data, multiviewica_delay


def test_estimated_delays_range():
    random_state = 42
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    treshold = 0.5
    delay_max = 10
    snr = 5  # Signal to noise ratio

    # Generate data
    _, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=0.,
        random_state=random_state)
    signal_power = np.mean(S ** 2)
    square_noise = signal_power / snr

    # Re generate data with noise
    X_list, _, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=square_noise,
        random_state=random_state)

    # Estimate delays with MVICAD
    _, _, _, tau_list, _ = multiviewica_delay(
        X_list, delay_max=delay_max, n_iter_delay=2, random_state=random_state)
    _, _, _, tau_list_1, _ = multiviewica_delay(
        X_list, delay_max=delay_max, n_iter_delay=2, optim_approach=1,
        random_state=random_state)
    _, _, _, tau_list_2, _ = multiviewica_delay(
        X_list, delay_max=delay_max, n_iter_delay=2, optim_approach=2,
        random_state=random_state)
    _, _, _, tau_list_with_f, _ = multiviewica_delay(
        X_list, delay_max=delay_max, n_iter_delay=2, optim_delays_with_f=True,
        random_state=random_state)

    outsiders = np.arange(delay_max+1, n-delay_max)
    assert np.sum(np.isin(tau_list, outsiders)) == 0
    assert np.sum(np.isin(tau_list_1, outsiders)) == 0
    assert np.sum(np.isin(tau_list_2, outsiders)) == 0
    assert np.sum(np.isin(tau_list_with_f, outsiders)) == 0

import numpy as np
from multiviewica_delay import generate_data, multiviewica_delay


def test_multiviewica_delay():
    random_state = 42
    m = 10  # number of subjects
    p = 5  # number of sources
    n = 400  # number of time points
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 10
    snr = 5  # Signal to noise ratio

    # Generate data
    _, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=0.,
        random_state=random_state)
    signal_power = np.mean(S ** 2)

    noise = np.sqrt(signal_power / snr)
    # Re generate data with noise
    X_list, _, true_tau_list, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise,
        random_state=random_state)

    # Estimate delays
    _, _, _, tau_list, tau_list_init = multiviewica_delay(
        X_list, n_iter_delay=1, random_state=random_state)

    np.testing.assert_array_equal(tau_list, true_tau_list)

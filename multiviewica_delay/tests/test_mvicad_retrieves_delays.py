import numpy as np
from multiviewica_delay import generate_data, multiviewica_delay


def normalize_delays(tau_list, n):
    tau_list -= tau_list[0]
    tau_list %= n
    return tau_list


def test_mvicad_retrieves_delays():
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
    X_list, _, true_tau_list, _, _ = generate_data(
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

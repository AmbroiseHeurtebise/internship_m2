import numpy as np
from multiviewica_delay.multiviewica_shifts._generate_data import generate_data
from multiviewica_delay import mvica_s


def normalize_delays(tau_list, n):
    tau_list -= tau_list[0]
    tau_list %= n
    return tau_list


def test_every_n_iter_delay():
    random_state = 10
    m = 2
    p = 3
    n = 50
    nb_intervals = 2
    nb_freqs = 5
    treshold = 0.5
    max_delay = 5
    snr = 8  # Signal to noise ratio

    # Generate data
    _, _, _, _, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=0.,
        random_state=random_state)
    signal_power = np.mean(S ** 2)
    square_noise = signal_power / snr

    # Re generate data with noise
    X_list, _, true_tau_list, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=max_delay, noise=square_noise,
        random_state=random_state)

    # Estimate delays with MVICAD
    results_dict = mvica_s(
        X_list,
        max_delay=max_delay,
        optim_delays_every_n_iter=10,
        random_state=random_state,
        shared_delays=True,
        return_every_iter=True,
    )
    print(results_dict["Delays"])
    tau_list = results_dict.iloc[-1]["Delays"]

    # Normalize delays
    true_tau_list = normalize_delays(true_tau_list, n)
    tau_list = normalize_delays(tau_list, n)

    # Test if arrays are equal
    np.testing.assert_array_equal(tau_list, true_tau_list)

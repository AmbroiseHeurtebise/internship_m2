import numpy as np
from multiviewica_delay import generate_data, _plot_delayed_sources, _apply_delay, multiviewica_delay


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 20
    noise = 10 ** 0.5  # 1e-6, 1, 10 ** 0.5
    random_state = 12

    # Generate data
    X_list, _, true_tau_list, S_list, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise,
        random_state=random_state)

    # True sources
    nb_plots = 4
    _plot_delayed_sources(S_list[:nb_plots], nb_intervals=nb_intervals)
    print("True delays : {}".format((true_tau_list - true_tau_list[0]) % n))

    # MVICAD
    W_permica, tau_list_permica, W_main, tau_list_main = multiviewica_delay(
        X_list, delay_max=delay_max, random_state=random_state,
        return_unmixing_delays_both_steps=True)
    tau_list = (tau_list_permica + tau_list_main) % n

    # Permica
    S_list_perm = np.array([np.dot(W, X) for W, X in zip(W_permica, X_list)])
    S_list_perm = _apply_delay(S_list_perm, -tau_list_permica)
    _plot_delayed_sources(
        S_list_perm[:nb_plots], height=0.25, nb_intervals=nb_intervals)
    print("Delays init : {}".format(tau_list_permica))

    # Main
    S_list_main = np.array([np.dot(W, X) for W, X in zip(W_main, X_list)])
    S_list_main = _apply_delay(S_list_main, -tau_list)
    _plot_delayed_sources(S_list_main[:nb_plots], nb_intervals=nb_intervals)
    print("Delays main : {}".format(tau_list_main))

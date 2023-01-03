import numpy as np
from multiviewica_delay import generate_data, permica, _plot_delayed_sources


if __name__ == '__main__':
    # Parameters
    m = 4
    p = 2
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    treshold = 1
    delay_max = 100
    noise = 0.5
    random_state = np.random.randint(1000)
    print(random_state)

    # Generate data
    X_list, _, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay_max, noise=noise, random_state=random_state)

    # PermICA
    _, W_list, _, _ = permica(X_list, delay_max=None)
    Y_list = np.array([np.dot(W, X) for W, X in zip(W_list, X_list)])

    # Print
    height = 0.5
    _plot_delayed_sources(Y_list, height, nb_intervals)

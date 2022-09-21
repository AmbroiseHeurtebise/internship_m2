import numpy as np
from delay_multiviewica import _plot_delayed_sources, generate_data, plot_sources


if __name__ == '__main__':
    # Parameters
    m = 4
    p = 2
    n = 400  # must be divisible by the number of intervals
    nb_intervals = 4
    nb_freqs = 20
    treshold = 3
    delay = 50
    noise = 0.1
    random_state = np.random.randint(1000)
    print(random_state)

    # Generate sources
    X_list, _, _, S_list, S = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay, noise=noise, random_state=random_state)

    # Plot
    plot_sources(S, nb_intervals)
    # _plot_delayed_sources(S_list)
    # _plot_delayed_sources(X_list)

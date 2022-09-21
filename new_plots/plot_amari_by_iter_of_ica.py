import numpy as np
import matplotlib.pyplot as plt
from picard import amari_distance
from delay_multiviewica import delay_multiviewica, multiviewica, create_sources_pierre, generate_data


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 2
    n = 400
    nb_intervals = 4
    nb_freqs = 5
    treshold = 3
    delay = 50
    noise = 0.2
    random_state = np.random.randint(1000)
    print(random_state)

    # Generate data
    # X_list, A_list, _ = create_sources_pierre(
    #     m, p, n, delay, sigma=noise, random_state=random_state)
    X_list, A_list, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=treshold, delay=delay, noise=noise, random_state=random_state)

    # Run ICA
    _, _, _, _, W_total_delay_mvica_earl_stop = delay_multiviewica(
        X_list, early_stopping_delay=10, random_state=random_state,
        return_basis_list=True)

    _, _, _, W_total_mvica = multiviewica(
        X_list, random_state=random_state, return_basis_list=True)

    # Compute Amari distance
    amari_mvica = np.array([np.sum(
        [amari_distance(W, A) for W, A in zip(W_list, A_list)])
        for W_list in W_total_mvica])

    amari_delay_mvica_earl_stop = np.array([np.sum(
        [amari_distance(W, A) for W, A in zip(W_list, A_list)])
        for W_list in W_total_delay_mvica_earl_stop])

    # Plot
    plt.plot(amari_delay_mvica_earl_stop, label='delay_mvica_earl_stop_10')
    plt.plot(amari_mvica, label='mvica')
    plt.title('Amari distance of mvica and delay_mvica; random state = {}'.format(random_state))
    plt.xlabel('Iterations')
    plt.ylabel('Amari distance')
    plt.legend()
    plt.grid()
    plt.savefig("new_figures/amari_by_iter_of_ica.pdf")
    plt.show()

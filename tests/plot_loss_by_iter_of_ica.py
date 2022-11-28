import matplotlib.pyplot as plt
from multiviewica_delay import multiviewica_delay, generate_data


def run_experiment(random_state):
    # Generate data
    X_list, _, _, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise, random_state=random_state)

    # Run ICA
    _, _, _, _, losses = multiviewica_delay(
        X_list, optim_delays_ica=False, random_state=random_state, return_loss=True)
    loss_total, loss_partial = losses

    return loss_total, loss_partial


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 10
    noise = 1
    random_state = 12

    # Results
    loss_total, loss_partial = run_experiment(random_state)

    # Plot
    # plt.plot(loss_total)
    # plt.title('MVICAD total loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Total loss')
    # plt.grid()
    # plt.savefig("tests_figures/total_loss_by_iter_of_ica.pdf")

    plt.plot(loss_partial)
    plt.title('MVICAD partial loss')
    plt.xlabel('Iterations')
    plt.ylabel('Partial loss')
    plt.grid()
    plt.savefig("tests_figures/partial_loss_by_iter_of_ica.pdf")

    plt.show()

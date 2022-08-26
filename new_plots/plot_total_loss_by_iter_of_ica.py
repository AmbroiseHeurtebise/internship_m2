import matplotlib.pyplot as plt
from delay_multiviewica import delay_multiviewica, multiviewica, create_sources_pierre


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 2
    n = 400
    delay = 10
    noise = 0.05
    random_state = 20

    # Generate data
    X_list, _, _ = create_sources_pierre(
        m, p, n, delay, sigma=noise, random_state=random_state)

    # Run ICA
    _, _, _, _, loss_total_delay_mvica_earl_stop = delay_multiviewica(
        X_list, early_stopping_delay=10, random_state=random_state, return_loss=True)

    _, _, _, loss_total_mvica = multiviewica(
        X_list, random_state=random_state, return_loss=True)
    
    # Plot
    plt.plot(loss_total_delay_mvica_earl_stop, label='delay_mvica_earl_stop_10')
    plt.plot(loss_total_mvica, label='mvica')
    plt.title('Total loss of mvica and delay_mvica; random state = {}'.format(random_state))
    plt.xlabel('Iterations')
    plt.ylabel('Total loss')
    plt.legend()
    plt.grid()
    plt.savefig("new_figures/total_loss_by_iter_of_ica.pdf")
    plt.show()

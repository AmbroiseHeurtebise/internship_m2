import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
from delay_multiviewica import _optimization_tau, _create_sources, _loss_delay


def run_experiment(n_sub, p, n, sources_noise, n_iter, random_state, init):
    # Generate data
    S_list, true_tau_list = _create_sources(
        n_sub, p, n, sources_noise, random_state)
    loss_true = _loss_delay(S_list, true_tau_list)

    # Optimization
    loss, _, _ = _optimization_tau(S_list, n_iter, init)
    loss = np.array(loss)

    output = {"Init": init, "random_state": random_state,
              "Loss": loss, "True loss": loss_true}
    return output


if __name__ == '__main__':
    # Parameters
    n_sub = 6  # must be even to plot the sources
    p = 2
    n = 400
    sources_noise = 0.5
    init = ['mean', 'first_subject']
    n_iter = 8
    nb_expe = 20
    random_states = range(nb_expe)
    N_JOBS = 4

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(n_sub, p, n, sources_noise, n_iter, r, i)
        for r, i
        in product(random_states, init)
    )
    results = pd.DataFrame(results)

    loss_mean = results[results["Init"] == 'mean']['Loss'].to_numpy()
    loss_first = results[results["Init"] == 'first_subject']['Loss'].to_numpy()
    loss_true = results[results["Init"] == 'mean']['True loss'].to_numpy()

    mean_loss_mean = np.mean(loss_mean, axis=0)
    mean_loss_first = np.mean(loss_first, axis=0)
    mean_loss_true = np.mean(loss_true)
    std_loss_mean = np.std(loss_mean, axis=0)
    std_loss_first = np.std(loss_first, axis=0)
    std_loss_true = np.std(loss_true)

    # Plot
    plt.plot(mean_loss_mean, label='Init. mean')
    plt.fill_between(range(n_iter), mean_loss_mean-std_loss_mean,
                     mean_loss_mean+std_loss_mean, alpha=.2)
    plt.plot(mean_loss_first, label='Init. first sub.')
    plt.fill_between(range(n_iter), mean_loss_first -
                     std_loss_first, mean_loss_first+std_loss_first, alpha=.2)
    plt.plot([mean_loss_true] * n_iter, label='True')
    plt.fill_between(range(n_iter), mean_loss_true-std_loss_true,
                     mean_loss_true+std_loss_true, alpha=.2)
    plt.legend()
    plt.yscale('log')
    plt.title("Partial loss optimization for 2 different initializations")
    plt.xlabel("Number of iterations")
    plt.ylabel("Partial loss")
    plt.savefig("figures/loss_optim_tau_init.pdf")
    plt.show()

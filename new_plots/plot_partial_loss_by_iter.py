import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
from delay_multiviewica import _optimization_tau, _create_sources, create_sources_pierre, _loss_delay, _optimization_tau_approach1, _optimization_tau_approach2


def run_experiment(n_sub, p, n, sources_noise, n_iter, random_state, init):
    # Generate data
    X_list, A_list, true_tau_list = create_sources_pierre(n_sub, p, n, delay_max=0.2*n, sigma=sources_noise, random_state=random_state)
    W_list = np.array([np.linalg.inv(A) for A in A_list])
    S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
    true_tau_list = -true_tau_list % n
    # S_list, true_tau_list = _create_sources(
    #     n_sub, p, n, sources_noise, random_state)
    loss_true = _loss_delay(S_list, true_tau_list)

    # Optimization
    if init == 'mean':
        loss, _, _ = _optimization_tau_approach1(S_list, n_iter)
    elif init == 'first_subject':
        loss, _, _ = _optimization_tau_approach2(S_list, n_iter)
    elif init == 'combine':
        loss, _, _ = _optimization_tau(S_list, n_iter)
    else:
        raise ValueError("init should mean, first_subject or combine")
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
    init = ['mean', 'first_subject', 'combine']
    n_iter = 8
    nb_expe = 5
    random_states = range(nb_expe)
    N_JOBS = 8

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(n_sub, p, n, sources_noise, n_iter, r, i)
        for r, i
        in product(random_states, init)
    )
    results = pd.DataFrame(results)

    loss_mean = results[results["Init"] == 'mean']['Loss'].to_numpy()
    loss_first = results[results["Init"] == 'first_subject']['Loss'].to_numpy()
    loss_combine = results[results["Init"] == 'combine']['Loss'].to_numpy()
    loss_true = results[results["Init"] == 'mean']['True loss'].to_numpy()

    mean_loss_mean = np.mean(loss_mean, axis=0)
    mean_loss_first = np.mean(loss_first, axis=0)
    mean_loss_combine = np.mean(loss_combine, axis=0)
    mean_loss_true = np.mean(loss_true)
    std_loss_mean = np.std(loss_mean, axis=0)
    std_loss_first = np.std(loss_first, axis=0)
    std_loss_combine = np.std(loss_combine, axis=0)
    std_loss_true = np.std(loss_true)

    # Plot
    plt.plot(mean_loss_mean, label='Init. mean')
    plt.fill_between(range(n_iter), mean_loss_mean-std_loss_mean,
                     mean_loss_mean+std_loss_mean, alpha=.2)
    plt.plot(mean_loss_first, label='Init. first sub.')
    plt.fill_between(range(n_iter), mean_loss_first -
                     std_loss_first, mean_loss_first+std_loss_first, alpha=.2)
    plt.plot(mean_loss_combine, label='Combine')
    plt.fill_between(range(n_iter), mean_loss_combine -
                     std_loss_combine, mean_loss_combine+std_loss_combine, alpha=.2)
    plt.plot([mean_loss_true] * n_iter, label='True')
    plt.fill_between(range(n_iter), mean_loss_true-std_loss_true,
                     mean_loss_true+std_loss_true, alpha=.2)
    plt.legend()
    plt.yscale('log')
    plt.title("Partial loss optimization for 2 different initializations")
    plt.xlabel("Number of iterations")
    plt.ylabel("Partial loss")
    plt.savefig("new_figures/partial_loss_by_iter.pdf")
    plt.show()

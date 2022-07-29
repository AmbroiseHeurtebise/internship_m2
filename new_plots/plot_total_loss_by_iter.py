import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from delay_multiviewica import create_sources_pierre, _Y_avg_init, _apply_delay, _new_delay_estimation


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))


def _total_loss_function(W_list, Y_list, Y_avg, noise):
    _, p, _ = W_list.shape
    loss = np.mean(_logcosh(Y_avg)) * p
    for (W, Y) in zip(W_list, Y_list):
        loss -= np.linalg.slogdet(W)[1]
        loss += 1 / (2 * noise) * np.mean((Y - Y_avg) ** 2) * p
    return loss


def _total_loss_delay(W_list, S_list, tau_list, noise):
    Y_list = _apply_delay(S_list, tau_list)
    Y_avg = np.mean(Y_list, axis=0)
    return _total_loss_function(W_list, Y_list, Y_avg, noise)


def _total_loss_delay_ref(W_list, S_list, tau_list, Y_avg, noise):
    Y_list = _apply_delay(S_list, tau_list)
    return _total_loss_function(W_list, Y_list, Y_avg, noise)


def _partial_loss_function(Y_list, Y_avg, noise):
    n_sub, p, _ = Y_list.shape
    loss = 1 / (2 * noise) * np.mean((Y_list - Y_avg) ** 2) * n_sub * p
    return loss


def _loss_delay(S_list, tau_list, noise):
    Y_list = _apply_delay(S_list, tau_list)
    Y_avg = np.mean(Y_list, axis=0)
    return _partial_loss_function(Y_list, Y_avg, noise)


def _loss_delay_ref(S_list, tau_list, Y_avg, noise):
    Y_list = _apply_delay(S_list, tau_list)
    return _partial_loss_function(Y_list, Y_avg, noise)


def _optimization_tau_true_loss(W_list, S_list, n_iter, init, noise=1):
    n_sub, _, n = S_list.shape
    Y_avg = _Y_avg_init(S_list, init)
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    partial_loss = []
    total_loss = []
    for iter in range(n_iter):
        partial_loss.append(_loss_delay_ref(S_list, tau_list, Y_avg, noise))
        total_loss.append(_total_loss_delay_ref(
            W_list, S_list, tau_list, Y_avg, noise))
        new_tau_list = np.zeros(n_sub, dtype=int)
        for i in range(n_sub):
            new_tau_list[i] = _new_delay_estimation(Y_list[i], Y_avg, n_sub)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([Y_list[i]], [new_tau_list[i]])
            # Y_avg = np.mean(Y_list, axis=0)
            # it doesn't make sense to initialize Y_avg at the sources of the first subject
            # because the formula of Y_avg changes at the second iteration
            # if we use Y_avg = np.mean(Y_list, axis=0)
            if init == "first_subject" and iter == 0 and i > 0:
                Y_avg += Y_list[i] / i
                Y_avg *= i / (i+1)
            if init == "mean" or iter > 0:
                Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list += new_tau_list
        tau_list %= n
    return partial_loss, total_loss, tau_list, Y_avg


def run_experiment(n_sub, p, n, sources_noise, n_iter, random_state, init):
    noise = 1
    # Generate data
    X_list, A_list, true_tau_list = create_sources_pierre(n_sub, p, n, delay_max=0.2*n, sigma=sources_noise, random_state=random_state)
    W_list = np.array([np.linalg.inv(A) for A in A_list])
    S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
    true_tau_list = -true_tau_list % n
    partial_loss_true = _loss_delay(S_list, true_tau_list, noise)
    total_loss_true = _total_loss_delay(W_list, S_list, true_tau_list, noise)

    # Optimization
    partial_loss, total_loss, _, _ = _optimization_tau_true_loss(
        W_list, S_list, n_iter, init)
    partial_loss = np.array(partial_loss)
    total_loss = np.array(total_loss)

    output = {"Init": init, "random_state": random_state,
              "Partial loss": partial_loss, "Partial loss true delays": partial_loss_true,
              "Total loss": total_loss, "Total loss true delays": total_loss_true}
    return output


if __name__ == '__main__':
    # Parameters
    n_sub = 6  # must be even to plot the sources
    p = 2
    n = 400
    sources_noise = 0.5
    init = 'mean'
    n_iter = 8
    nb_expe = 50
    random_states = range(nb_expe)
    N_JOBS = 4

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(n_sub, p, n, sources_noise, n_iter, r, init)
        for r in random_states
    )
    results = pd.DataFrame(results)

    partial_loss = results['Partial loss'].to_numpy()
    partial_loss_true = results['Partial loss true delays'].to_numpy()
    total_loss = results['Total loss'].to_numpy()
    total_loss_true = results['Total loss true delays'].to_numpy()

    mean_partial_loss = np.mean(partial_loss, axis=0)
    mean_partial_loss_true = np.mean(partial_loss_true)
    # std_partial_loss = np.std(partial_loss, axis=0)
    # std_partial_loss_true = np.std(partial_loss_true)

    mean_total_loss = np.mean(total_loss, axis=0)
    mean_total_loss_true = np.mean(total_loss_true)
    # std_total_loss = np.std(total_loss, axis=0)
    # std_total_loss_true = np.std(total_loss_true)

    # Plot
    plt.plot(mean_partial_loss, label='Partial loss')
    # plt.fill_between(range(n_iter), mean_partial_loss-std_partial_loss,
    #                  mean_partial_loss+std_partial_loss, alpha=.2)
    plt.plot([mean_partial_loss_true] * n_iter,
             label='Partial loss true delays')
    # plt.fill_between(range(n_iter), mean_partial_loss_true-std_partial_loss_true,
    #                  mean_partial_loss_true+std_partial_loss_true, alpha=.2)

    plt.plot(mean_total_loss, label='Total loss')
    # plt.fill_between(range(n_iter), mean_total_loss-std_total_loss,
    #                  mean_total_loss+std_total_loss, alpha=.2)
    plt.plot([mean_total_loss_true] * n_iter, label='Total loss true delays')
    # plt.fill_between(range(n_iter), mean_total_loss_true-std_total_loss_true,
    #                  mean_total_loss_true+std_total_loss_true, alpha=.2)

    plt.legend()
    # plt.yscale('log')
    plt.title("True loss and partial loss from partial loss optimization")
    plt.xlabel("Number of iterations")
    plt.ylabel("True loss and partial loss")
    plt.savefig("new_figures/total_loss_by_iter.pdf")
    plt.show()

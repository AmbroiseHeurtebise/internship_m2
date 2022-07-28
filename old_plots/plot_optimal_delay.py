import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def amari_distance(P):
    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)
    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


if __name__ == '__main__':
    # Parameters
    max_delay = 35
    stepsize = 2
    delay_abs = np.arange(-max_delay+stepsize, max_delay, stepsize)

    # Load sources
    savefile_name1 = "data/sources1.npy"
    savefile_name2 = "data/sources2.npy"
    if os.path.isfile(savefile_name1) and os.path.isfile(savefile_name2):
        S1 = np.load(savefile_name1)
        S2 = np.load(savefile_name2)
    else:
        print("No sources available \n")

    # Covariance matrices
    n_sources, n_samples = S1.shape
    neg_delay = delay_abs[delay_abs < 0]
    pos_delay = delay_abs[delay_abs >= 0]

    cov1 = np.array([np.dot(S1[:, delay:n_samples-max_delay+delay], S2[:, :n_samples-max_delay].T) / np.outer(
        np.std(S1[:, delay:n_samples-max_delay+delay], axis=1), np.std(S2[:, :n_samples-max_delay], axis=1)) for delay in -neg_delay])
    cov2 = np.array([np.dot(S1[:, :n_samples-max_delay], S2[:, delay:n_samples-max_delay+delay].T) / np.outer(
        np.std(S1[:, :n_samples-max_delay], axis=1), np.std(S2[:, delay:n_samples-max_delay+delay], axis=1)) for delay in pos_delay])
    cov = np.concatenate((cov1, cov2))

    # Amari distance of the covariance matrices
    distances = np.array([amari_distance(c) for c in cov])

    # Plot
    plt.plot(delay_abs, distances)
    plt.title("Amari distance of the cov. matrices shifted by a delay", fontweight="bold")
    plt.xlabel("Delay")
    plt.ylabel("Amari distance")
    plt.savefig("figures/optimal_delay.pdf")

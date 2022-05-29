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
    max_delay = 50
    stepsize = 5
    delay_abs = np.arange(-max_delay+stepsize, max_delay, stepsize)

    # Get the sources
    savefile_name = "mvica_datasets.pkl"
    if os.path.isfile(savefile_name):
        with open(savefile_name, "rb") as save_ica_file:
            S1, S2 = pickle.load(save_ica_file)
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
    plt.savefig("optimal_delay.pdf")
    plt.show()

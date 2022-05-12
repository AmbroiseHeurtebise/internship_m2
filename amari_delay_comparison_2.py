import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from joblib import Parallel, delayed
from picard import amari_distance
from multiviewica import multiviewica
from multiviewica import groupica


def create_sources_gauss(p, n, mu, var, window_length, peaks_height, sources_height, noise):
    """
    Create random sources using Gaussian r.v. to model the interval between two successive peaks. 

    Parameters
    ----------
    p : int
        Number of sources.
    n : int
        Number of samples.
    mu : np array of shape (p, )
        Expectation vector of the activation times.
    var : np array of shape (p, )
        Variance of the activation times of each source.
    window_length : int (odd)
        Size of the Hamming window.
    peaks_height : int
        Represents the hight of peaks divided by the height of the cruve outside the peaks.
    sources_height : np array of shape (p, )
        Height of each source.
    """
    # start allows the sources not to begin in 0
    start = np.outer(np.random.rand(p) * math.pi / 2, np.ones(n))
    # scale: s(t) = sin(scale * t)
    scale = np.diag(np.random.rand(p) + 0.5) / 10
    S = np.sin(start + np.dot(scale, np.outer(np.ones((p, 1)), np.arange(n))))
    window = peaks_height * np.hamming(window_length)
    half_window_length = int((window_length - 1) / 2)
    act_length = int(2*n/np.min(mu))
    # act is a matrix of size (p, int(2*n/np.min(mu)) which contains the activation times of the sources
    act = np.random.randn(p, act_length)
    act = np.dot(np.diag(var), act)
    act += np.outer(mu, np.ones(act_length))
    act = np.cumsum(act, axis=1).astype(int)
    act_max = np.max(act)
    # new_act is a matrix of size (p, n) which is positive when activation and 0 otherwise
    new_act = np.zeros((p, act_max+1))
    for i in range(p):
        new_act[i][act[i]] = 1
    new_act = new_act[:, :n]
    for i in range(p):
        for j in range(half_window_length, n-half_window_length):
            if(new_act[i, j] == 1):
                new_act[i, j-half_window_length: j+half_window_length+1] = window
    S *= 1 + new_act
    S = np.dot(np.diag(sources_height), S)
    add_noise = np.dot(np.diag(noise), np.random.randn(p, n))
    S += add_noise
    return S


def create_model(n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise):
    max_delay = np.max(delay)
    S = create_sources_gauss(p, n, mu, var, window_length, peaks_height, sources_height, noise)
    A = np.random.randn(n_subjects, p, p)
    N = np.random.randn(n_subjects, p, n - max_delay)
    # X is a matrix of size (p, n - max_delay)
    X = np.array([A[i].dot(S[:, delay[i]:n - max_delay + delay[i]] + sigma * N[i]) for i in np.arange(n_subjects)])
    return X, A, S, N


def initialize_model(n_subjects, p, sup_delay):
    # Parameters
    sigma = 0.1
    delay = np.random.randint(0, sup_delay, n_subjects)
    mu = 100 * np.random.randn(p) + 400
    mu[mu < 5] = 5  # treshold
    var = np.random.exponential(scale=50, size=p)
    window_length = 99  # must be odd
    peaks_height = 10
    sources_height = 2 * np.random.rand(p) + 0.5
    noise = np.random.rand(p) / 2 + 0.1
    return sigma, delay, mu, var, window_length, peaks_height, sources_height, noise


def univiewica(X):
    n_subjects, p, n = X.shape
    W_approx = []
    for i in range(n_subjects):
        _, W, _ = multiviewica(X[i].reshape(1, p, n))
        W_approx.append(W.reshape(p, p))
    W_approx = np.asarray(W_approx)
    return W_approx


def loop_over_expe(nb_expe, n_subjects, p, n, sup_delay):
    mvica_amari = []
    groupica_amari = []
    uniica_amari = []

    for i in range(nb_expe):
        # Initialization
        sigma, delay, mu, var, window_length, peaks_height, sources_height, noise = initialize_model(n_subjects, p, sup_delay)

        # Create model
        X, A, _, _ = create_model(n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise)

        # MVICA
        _, W_approx, _ = multiviewica(X)
        amari_distances = [amari_distance(w, a) for (w, a) in zip(W_approx, A)]
        mvica_amari.append(np.mean(amari_distances))

        # GroupICA
        _, W_approx, _ = groupica(X)
        amari_distances = [amari_distance(w, a) for (w, a) in zip(W_approx, A)]
        groupica_amari.append(np.mean(amari_distances))

        # UniICA
        W_approx = univiewica(X)
        amari_distances = [amari_distance(w, a) for (w, a) in zip(W_approx, A)]
        uniica_amari.append(np.mean(amari_distances))

    return (mvica_amari, groupica_amari, uniica_amari)


def plot_distances(distances):
    eps = 1e-4

    # Mean and std
    mean_amari = np.mean(distances, axis=2)
    std_amari = np.std(distances, axis=2)

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Quantity of delay')
    ax.set_ylabel('Amari distance')

    ax.plot(delay-1, mean_amari[:, 0], label='MVICA', c='r')
    low_curve = mean_amari[:, 0] - std_amari[:, 0]
    ax.fill_between(delay-1, low_curve * (low_curve > 0) + eps * (low_curve <= 0), mean_amari[:, 0] + std_amari[:, 0], color='r', alpha=.1)
    ax.plot(delay-1, mean_amari[:, 1], label='GroupICA', color='b')
    low_curve = mean_amari[:, 1] - std_amari[:, 1]
    ax.fill_between(delay-1, low_curve * (low_curve > 0) + eps * (low_curve <= 0), mean_amari[:, 1] + std_amari[:, 1], color='b', alpha=.1)
    ax.plot(delay-1, mean_amari[:, 2], label='UniICA', color='orange')
    low_curve = mean_amari[:, 2] - std_amari[:, 2]
    ax.fill_between(delay-1, low_curve * (low_curve > 0) + eps * (low_curve <= 0), mean_amari[:, 2] + std_amari[:, 2], color='orange', alpha=.1)

    ax.set_yscale('log')
    ax.legend()
    plt.title('Amari distance wrt the quantity of delay', fontweight="bold")
    plt.show()


if __name__ == '__main__':
    n_subjects = 8
    p = 6
    n = 2000
    delay = np.arange(1, 25)
    nb_expe = 15

    # Loop over delay
    distances = Parallel(n_jobs=4)(delayed(loop_over_expe)(nb_expe, n_subjects, p, n, d) for d in tqdm(delay))

    # Plot
    plot_distances(distances)

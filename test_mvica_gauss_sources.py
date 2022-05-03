# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from picard import amari_distance
from multiviewica import multiviewica

# %%
def create_sources_gauss(p, n, mu, var, window_length, peaks_height, sources_height, noise):
    """
    Create random sources.

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
    scale = np.diag(np.random.rand(p) + 0.5)
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


def initialize_model(sup_delay):
    # Parameters
    n_subjects = 20
    p = 3
    n = 200
    sigma = 0.1
    delay = np.random.randint(0, sup_delay, n_subjects)
    mu = 10 * np.random.randn(p) + 40  # np.repeat(40, p)
    mu[mu<5] = 5  # treshold
    var = np.random.exponential(scale=5, size=p)  # np.repeat(1, p)
    window_length = 7  # must be odd
    peaks_height = 10
    sources_height = 2 * np.random.rand(p) + 0.5
    noise = np.random.rand(p) + 0.5
    return n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise


def plot_sources(S):
    # Show sources
    for i in range(len(S)):
        plt.plot(S[i] + 25 * i)
    plt.title('Sources of the model')
    plt.xlabel('Samples')
    plt.yticks([])
    plt.savefig('sources.pdf')
    plt.show()


def ICA_amari(X, A):
    # ICA
    start = time.time()
    _, W_approx, S_approx = multiviewica(X)
    end = time.time()
    print("ICA takes {:.2f} s".format(end - start))
    # Amari distances
    amari_distances = [amari_distance(w, a) for (w, a) in zip(W_approx, A)]
    # n_subjects = len(X)
    # print("Amari distances for the {} subjects : {}".format(n_subjects, amari_distances))
    return W_approx, S_approx, amari_distances


def plot_permutation_matrices(W_approx, A, amari_distances):
    n_subjects = len(A)
    # Plots of W^i A^i
    for (i, w, a) in zip(np.arange(n_subjects), W_approx, A):
        plt.figure()
        plt.matshow(np.abs(np.dot(w, a)))
        dist = amari_distances[i]
        plt.title("Amari distance = {:.5f}".format(dist))
    # idea: plots must represent the same permutation matrix 
    # if there is a delay, these matrices are different, hence the bad reconstruction
    # yet, the Amari distances are low (normally)


def plot_correlation_S(S, S_approx, delay):
    max_delay = np.max(delay)
    n = np.shape(S)[1]
    # Plot of correlation matrix S_approx S.T
    abs_cov = np.abs(S_approx.dot(S[:, :n - max_delay].T))
    plt.matshow(abs_cov / np.max(abs_cov))
    plt.title("Correlation betxeen S and S_approx")
    plt.colorbar()


def loop_average_amari_distance(sup_delay=1, verbose=False):
    """
    Create a model and compute the average Amari distance between A_i and W_i. 

    Parameters
    ----------
    sup_delay : int, >0
        Quantity of delay between subjects. If 1, there is no delay. 
    verbose : bool
        In order to plot the sources and the permutation matrices or not. 
    """
    # Initialization
    n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise = initialize_model(sup_delay)

    # Create model
    start = time.time()
    X, A, S, N = create_model(n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise)
    end = time.time()
    print("Creating the model takes {:.2f} s".format(end - start))

    # ICA and Amari distance
    W_approx, S_approx, amari_distances = ICA_amari(X, A)
    mean_amari_distances = np.mean(amari_distances)

    if verbose:
        # Plot sources
        plot_sources(S)

        # Permutation matrices
        plot_permutation_matrices(W_approx, A, amari_distances)
        plot_correlation_S(S, S_approx, delay)

    return mean_amari_distances

# %%
amari_distance_delay = []
delay = np.arange(1, 15)
for i in delay:
    mean_amari_distance = loop_average_amari_distance(sup_delay=i)
    amari_distance_delay.append(mean_amari_distance)
    print("Average Amari distance with delay {} : {:.5f} \n".format(i, np.mean(mean_amari_distance)))

plt.plot(amari_distance_delay)
plt.title("Average Amari distance wrt the quantity of delay")
plt.xlabel('Quantity of delay')
plt.ylabel('Average Amari distance')
plt.savefig('amar_dist2.pdf')
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
import math


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
    scale = np.diag(np.random.rand(p) + 0.5) / 20
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
    mu = 200 * np.random.randn(p) + 800
    mu[mu < 5] = 5  # treshold
    var = np.random.exponential(scale=100, size=p)
    window_length = 199  # must be odd
    peaks_height = 10
    sources_height = 2 * np.random.rand(p) + 0.5
    noise = np.random.rand(p) / 2 + 0.1
    return sigma, delay, mu, var, window_length, peaks_height, sources_height, noise


def plot_sources(S):
    for i in range(len(S)):
        plt.plot(S[i] + 25 * i)
    plt.title('Sources of the model')
    plt.xlabel('Samples')
    plt.yticks([])
    plt.savefig('sources.pdf')
    plt.show()


if __name__ == '__main__':
    n_subjects = 10
    p = 4
    n = 2000

    # Initialization
    sigma, delay, mu, var, window_length, peaks_height, sources_height, noise = initialize_model(n_subjects, p, sup_delay=1)

    # Create model
    _, _, S, _ = create_model(n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise)

    # Plot sources
    plot_sources(S)

import math
from itertools import product
import pickle

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, Memory
from sklearn.utils import check_random_state

from picard import amari_distance
from multiviewica import multiviewica
from multiviewica import groupica


def create_sources_gauss(p, n, mu, var, window_length, peaks_height, sources_height, noise,
                         random_state=None):
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
    random_state : int or RandomState or None
        The random number generator used for the initialization. If an integer,
        it is used to seed the random number generator. If None, use the global
        state.
    
    Returns
    -------
    XXX
    """
    rng = check_random_state(random_state)
    # start allows the sources not to begin in 0
    start = np.outer(rng.rand(p) * math.pi / 2, np.ones(n))
    # scale: s(t) = sin(scale * t)
    scale = np.diag(rng.rand(p) + 0.5) / 20
    S = np.sin(start + np.dot(scale, np.outer(np.ones((p, 1)), np.arange(n))))
    window = peaks_height * np.hamming(window_length)
    half_window_length = int((window_length - 1) / 2)
    act_length = int(2*n/np.min(mu))
    # act is a matrix of size (p, int(2*n/np.min(mu)) which contains the activation times of the sources
    act = rng.randn(p, act_length)
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
    add_noise = np.dot(np.diag(noise), rng.randn(p, n))
    S += add_noise
    return S


def create_model(n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height, sources_height, noise,
                 random_state=None):
    """XXX"""
    rng = check_random_state(random_state)
    max_delay = np.max(delay)
    S = create_sources_gauss(p, n, mu, var, window_length, peaks_height, sources_height, noise)
    A = rng.randn(n_subjects, p, p)
    N = rng.randn(n_subjects, p, n - max_delay)
    # X is a matrix of size (p, n - max_delay)
    X = np.array([A[i].dot(S[:, delay[i]:n - max_delay + delay[i]] + sigma * N[i]) for i in range(n_subjects)])
    return X, A, S, N


def initialize_model(n_subjects, p, sup_delay, random_state=None):
    # Parameters
    rng = check_random_state(random_state)
    sigma = 0.1
    delay = rng.randint(0, sup_delay, n_subjects)
    mu = 200 * rng.randn(p) + 800
    mu[mu < 5] = 5  # treshold
    var = rng.exponential(scale=100, size=p)
    window_length = 199  # must be odd
    peaks_height = 10
    sources_height = 2 * rng.rand(p) + 0.5
    noise = rng.rand(p) / 2 + 0.1
    return sigma, delay, mu, var, window_length, peaks_height, sources_height, noise


def univiewica(X, random_state):
    n_subjects, p, n = X.shape
    W_approx = []
    for i in range(n_subjects):
        _, W, _ = multiviewica(X[i].reshape(1, p, n), random_state=random_state)
        W_approx.append(W.reshape(p, p))
    W_approx = np.asarray(W_approx)
    return W_approx


mem = Memory('.')


@mem.cache
def run_experiment(algo_name, n_subjects, p, n, sup_delay, random_state):
    rng = check_random_state(random_state)
    # Initialization
    sigma, delay, mu, var, window_length, peaks_height, sources_height, noise = \
        initialize_model(n_subjects, p, sup_delay, random_state=rng)

    # Create model
    X, A, S, _ = create_model(
        n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height,
        sources_height, noise, random_state=rng
    )

    # ICA
    if(algo_name == 'MVICA'):
        _, W_approx, _ = multiviewica(X, random_state=random_state)
    elif(algo_name == 'GroupICA'):
        _, W_approx, _ = groupica(X, random_state=random_state)
    elif(algo_name == 'UniviewICA'):
        W_approx = univiewica(X, random_state=random_state)

    # Amari distance
    amari_distances = [amari_distance(w, a) for (w, a) in zip(W_approx, A)]
    mean_amari_distances = np.mean(amari_distances)

    # Output
    output = {"Algo": algo_name, "Delay": sup_delay, "random_state": random_state, "Sources": S,
              "Mixing": A, "Unmixing": W_approx, "Amari": amari_distances,
              "Mean_Amari": mean_amari_distances}
    return output


if __name__ == '__main__':
    # Parameters
    # N_JOBS = 10
    N_JOBS = 60
    algos_name = ['MVICA', 'GroupICA', 'UniviewICA']
    n_subjects = 15
    n_sources = 6
    n_samples = 3500
    sup_delay = np.arange(1, 50, 5)
    nb_expe = 50
    random_states = np.arange(nb_expe)
    # random_states = np.random.choice(1000, nb_expe, replace=False)

    # Run experiments in parallel with cartesian product on all parameters
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(a, n_subjects, n_sources, n_samples, d, r) 
        for a, d, r
        in product(algos_name, sup_delay, random_states)
    )
    results = pd.DataFrame(results)

    # Save results in a csv file
    with open("results_ica", "wb") as save_results_file:
        pickle.dump(results, save_results_file)

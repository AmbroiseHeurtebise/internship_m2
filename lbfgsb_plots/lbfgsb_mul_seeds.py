import numpy as np
import pandas as pd
import os
import time
import scipy
from sklearn.utils import check_random_state

import jax
import jax.numpy as jnp
import jaxopt

from picard import amari_distance

from multiviewica_delay import (
    _apply_continuous_delays,
    generate_data,
    data_generation,
    data_generation_pierre,
    multiviewica_delay,
)


def _logcosh(X):
    Y = jnp.abs(X)
    return Y + jnp.log1p(jnp.exp(-2 * Y))


def optimize_unmixing_delays(
    W_list,
    delays,
    X_list,
    max_delay=10,
    noise=1,
    shared_delays=False,
    maxiter=1000,
    verbose=False,
):
    if verbose:
        start = time.time()
    m, p, _ = W_list.shape

    def loss_function(W_delays):
        W_list, delays = W_delays[:m*p**2].reshape((m, p, p)), W_delays[m*p**2:]
        if not shared_delays:
            delays = delays.reshape((m, p))
        S_list = jnp.array([jnp.dot(W, X) for W, X in zip(W_list, X_list)])
        Y_list = _apply_continuous_delays(
            S_list=S_list,
            tau_list=-delays,
            shared_delays=shared_delays,
            multiple_subjects=True,
            use_jax=True,
        )
        Y_avg = jnp.mean(Y_list, axis=0)
        loss = jnp.mean(_logcosh(Y_avg)) * p
        for W, Y in zip(W_list, Y_list):
            loss -= jnp.linalg.slogdet(W)[1]
            loss += 1 / (2 * noise) * jnp.mean((Y - Y_avg) ** 2) * p
        return loss

    init_params = jnp.concatenate([jnp.ravel(W_list), jnp.ravel(delays)])
    bounds_W_bias = (
        -np.ones_like(init_params) * np.inf,
        np.ones_like(init_params) * np.inf)
    if shared_delays:
        bounds_W_bias[0][-m:] = -max_delay
        bounds_W_bias[1][-m:] = max_delay
    else:
        bounds_W_bias[0][-m*p:] = -max_delay
        bounds_W_bias[1][-m*p:] = max_delay
    solver = jaxopt.LBFGSB(fun=loss_function, maxiter=maxiter)
    res = solver.run(init_params, bounds=bounds_W_bias)
    params, state = res
    W_list_final, delays_final = params[:m*p**2].reshape((m, p, p)), params[m*p**2:]
    if not shared_delays:
        delays_final = delays_final.reshape((m, p))
    if verbose:
        end = time.time()
        computation_time = end - start
        return W_list_final, delays_final, state, computation_time
    return W_list_final, delays_final


def optimize_unmixing_alone(
    W_list,
    X_list,
    noise=1,
    maxiter=1000,
    verbose=False,
):
    if verbose:
        start = time.time()
    m, p, _ = W_list.shape

    def loss_function(W_list):
        W_list = W_list.reshape((m, p, p))
        S_list = jnp.array([jnp.dot(W, X) for W, X in zip(W_list, X_list)])
        S_avg = jnp.mean(S_list, axis=0)
        loss = jnp.mean(_logcosh(S_avg)) * p
        for W, S in zip(W_list, S_list):
            loss -= jnp.linalg.slogdet(W)[1]
            loss += 1 / (2 * noise) * jnp.mean((S - S_avg) ** 2) * p
        return loss

    init_params = jnp.ravel(W_list)
    solver = jaxopt.LBFGS(fun=loss_function, maxiter=maxiter)
    res = solver.run(init_params)
    params, state = res
    W_list_final = params.reshape((m, p, p))

    if verbose:
        end = time.time()
        computation_time = end - start
        return W_list_final, state, computation_time
    return W_list_final


def generate_X_A_delays_S(generation_function, random_state=None):
    if generation_function == "first":
        X_list, A_list, true_tau_list, S_list, S = generate_data(
            m=m,
            p=p,
            n=n,
            nb_intervals=nb_intervals,
            nb_freqs=nb_freqs,
            treshold=treshold,
            delay=max_delay,
            noise=noise,
            random_state=random_state,
            shared_delays=shared_delays
        )
    elif generation_function == "second":
        X_list, A_list, true_tau_list, S_list, S = data_generation(
            m=m,
            p=p,
            n=n,
            max_delay=max_delay,
            noise=noise,
            shared_delays=shared_delays,
            random_state=random_state,
            n_concat=n_concat,
        )
    else:
        X_list, A_list, true_tau_list, S_list, S = data_generation_pierre(
            n_subjects=m,
            n_sources=p,
            n_bins=n_bins,
            n_samples_per_interval=n_samples_per_interval,
            freq_level=freq_level,
            max_delay=max_delay,
            noise=noise,
            shared_delays=shared_delays,
            random_state=random_state,
        )
    return X_list, A_list, true_tau_list, S_list, S


# estimated sources are not necessarily in the same order as true sources
def find_order(S1, S2):
    S1 = S1 / np.linalg.norm(S1, axis=1, keepdims=True)
    S2 = S2 / np.linalg.norm(S2, axis=1, keepdims=True)
    M = np.abs(np.dot(S1, S2.T))
    try:
        _, order = scipy.optimize.linear_sum_assignment(-abs(M))
    except ValueError:
        order = np.arange(p)
    return order


def run_experiment(random_state):
    # generate data
    X_list, A_list, true_delays, S_list, S = generate_X_A_delays_S(
        generation_function, random_state)

    # initialize W_list, S_list and delays
    W_list_true = jnp.array([jnp.linalg.inv(A) for A in A_list])
    W_list_noise = jax.random.normal(jax.random.PRNGKey(random_state), (m, p, p))
    W_list_init = W_list_true + noise_level_init * W_list_noise
    if shared_delays:
        delays_init = jnp.zeros(m)
    else:
        delays_init = jnp.zeros((m, p))

    # L-BFGS-B
    W_lbfgsb, delays_lbfgsb, _, time_lbfgsb = optimize_unmixing_delays(
        W_list=W_list_init,
        delays=delays_init,
        X_list=X_list,
        shared_delays=shared_delays,
        max_delay=max_delay,
        maxiter=1000,
        verbose=True,
    )
    S_lbfgsb = np.mean([np.dot(W, X) for W, X in zip(W_lbfgsb, X_list)], axis=0)

    # L-BFGS-B without delays
    W_lbfgs_without_delay, _, time_lbfgs_without_delay = optimize_unmixing_alone(
        W_list=W_list_init,
        X_list=X_list,
        maxiter=1000,
        verbose=True,
    )

    # MVICAD with discrete delays
    start_mvicad = time.time()
    _, W_mvicad, S_mvicad, _, delays_mvicad, _ = multiviewica_delay(
        X_list,
        init=np.array(W_list_init),
        shared_delays=shared_delays,
        max_delay=max_delay,
        random_state=random_state,
        continuous_delays=False,
    )
    end_mvicad = time.time()
    time_mvicad = end_mvicad - start_mvicad
    delays_mvicad[delays_mvicad > n // 2] -= n

    # Amari distance
    amari_lbfgsb = np.mean([amari_distance(W, A) for W, A in zip(W_lbfgsb, A_list)])
    amari_mvicad = np.mean([amari_distance(W, A) for W, A in zip(W_mvicad, A_list)])
    amari_lbfgs_without_delay = np.mean(
        [amari_distance(W, A) for W, A in zip(W_lbfgs_without_delay, A_list)])
    rng = check_random_state(random_state)
    nb_seeds_random = 20
    amari_random = 0
    for _ in range(nb_seeds_random):
        W_rand = rng.randn(m, p, p)
        amari_random += np.mean([amari_distance(W, A) for W, A in zip(W_rand, A_list)])
    amari_random /= nb_seeds_random

    # estimated sources order
    if not shared_delays:
        order_lbfgsb = find_order(S, S_lbfgsb)
        delays_lbfgsb = delays_lbfgsb[:, order_lbfgsb]
        order_mvicad = find_order(S, S_mvicad)
        delays_mvicad = delays_mvicad[:, order_mvicad]

    # delay error
    delay_error_lbfgsb = np.mean(np.abs(delays_lbfgsb - true_delays))
    delay_error_mvicad = np.mean(np.abs(delays_mvicad - true_delays))
    delay_error_random = 0
    for _ in range(nb_seeds_random):
        delays_rand = rng.randint(low=-max_delay, high=max_delay, size=np.shape(true_delays))
        delay_error_random += np.mean(np.abs(delays_rand - true_delays))
    delay_error_random /= nb_seeds_random

    # result
    dict_res = {"amari_lbfgsb": [amari_lbfgsb],
                "amari_lbfgs_without_delay": [amari_lbfgs_without_delay],
                "amari_mvicad": [amari_mvicad],
                "amari_random": [amari_random],
                "time_lbfgsb": [time_lbfgsb],
                "time_lbfgs_without_delay": [time_lbfgs_without_delay],
                "time_mvicad": [time_mvicad],
                "delays_lbfgsb": [delays_lbfgsb],
                "delays_mvicad": [delays_mvicad],
                "true_delays": [true_delays],
                "delay_error_lbfgsb": [delay_error_lbfgsb],
                "delay_error_mvicad": [delay_error_mvicad],
                "delay_error_random": [delay_error_random]}
    return dict_res


if __name__ == '__main__':
    # params
    m = 5
    p = 2
    n = 600
    max_delay = 10
    noise_level_init = 0.1
    shared_delays = False
    nb_seeds = 20
    generation_function = "first"
    if generation_function == "first":
        nb_intervals = 5
        nb_freqs = 10
        treshold = 1
        noise = 0.8
    elif generation_function == "second":
        n_concat = 1
        noise = 0.01
    else:
        n_bins = 10
        n_samples_per_interval = n // n_bins
        freq_level = 40
        noise = 0.2

    # run expe
    df_res = pd.DataFrame()
    for random_state in range(nb_seeds):
        dict_expe = run_experiment(random_state)
        df_expe = pd.DataFrame(dict_expe)
        df_res = pd.concat([df_res, df_expe], ignore_index=True)

    # save dataframe
    results_dir = "/storage/store2/work/aheurteb/mvicad/lbfgsb_results/"
    output_dir = results_dir + f"{nb_seeds}_seeds"
    os.makedirs(output_dir, exist_ok=True)

    col_2d_arrays = ['delays_lbfgsb', 'delays_mvicad', 'true_delays']
    for col in col_2d_arrays:
        for i, arr in enumerate(df_res[col]):
            np.save(os.path.join(output_dir, f"{col}_array_{i}.npy"), arr)

    save_name = f"lbfgsb_benchmark_{nb_seeds}_seeds.csv"
    save_path = results_dir + save_name
    df_res.to_csv(save_path, index=False)

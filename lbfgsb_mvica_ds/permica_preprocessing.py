import numpy as np
import scipy
from tqdm import tqdm
from multiviewica_delay import _apply_delay_by_source, _apply_delay_one_source_or_sub


# function from multiviewica_delay/_permica.py
def _hungarian(M):
    u, order = scipy.optimize.linear_sum_assignment(-abs(M))
    vals = M[u, order]
    cost = np.sum(np.abs(vals))
    return order, np.sign(vals), cost


# function that finds the optimal delays; useful to find sources' order
def find_delays_permica(S_list_init, max_delay, delay_step=10):
    m, p, n = S_list_init.shape
    allowed_delays = np.concatenate(
        (np.arange(0, max_delay+1, delay_step), np.arange(n, n-max_delay, -delay_step)[1:][::-1]))
    allowed_delays_grid = np.array(np.meshgrid(*[allowed_delays] * p))
    allowed_delays_grid = allowed_delays_grid.reshape((p, -1))  # (p, n_delays_tested ** p)
    S = S_list_init[0].copy()
    optimal_delays = np.zeros((m, p), dtype="int")
    for i, s in enumerate(S_list_init[1:]):
        optimal_cost = -np.inf
        for delays in tqdm(allowed_delays_grid.T):
            s_delayed = _apply_delay_one_source_or_sub(s, -delays)
            M = np.dot(S, s_delayed.T)
            _, _, cost = _hungarian(M)
            if cost > optimal_cost:
                optimal_cost = cost
                optimal_delays[i+1] = delays
    S_list_init_delayed = _apply_delay_by_source(S_list_init, -optimal_delays)
    return optimal_delays, S_list_init_delayed


# function from permica that finds sources' order for each subject
def _find_ordering(S_list, n_iter=10):
    n_pb, p, _ = S_list.shape
    for i in range(len(S_list)):
        S_list[i] /= np.linalg.norm(S_list[i], axis=1, keepdims=1)
    S = S_list[0].copy()
    order = np.arange(p)[None, :] * np.ones(n_pb, dtype=int)[:, None]
    signs = np.ones_like(order)
    for _ in range(n_iter):
        for i, s in enumerate(S_list[1:]):
            M = np.dot(S, s.T)
            order[i + 1], signs[i + 1], _ = _hungarian(M)
        S = np.zeros_like(S)
        for i, s in enumerate(S_list):
            S += signs[i][:, None] * s[order[i]]
        S /= n_pb
    return order, signs, S


# same order for S_list and S_list_init
def find_order(S1, S2):
    p, n = S1.shape
    S1 = S1 / np.linalg.norm(S1, axis=1, keepdims=True)
    S2 = S2 / np.linalg.norm(S2, axis=1, keepdims=True)
    M = np.abs(np.dot(S1, S2.T))
    try:
        _, order = scipy.optimize.linear_sum_assignment(-abs(M))
    except ValueError:
        order = np.arange(p)
    return order


# find sign from sources' height
def find_sign(S_list):
    return np.array([[np.sign(np.max(s) + np.min(s)) for s in S] for S in S_list])


# # find sign by comparing to first subject
# def find_sign_first_sub(S_list):
#     _, p, n = S_list.shape
#     signs_sub_0 = np.array([np.sign(np.max(s) + np.min(s)) for s in S_list[0]])
#     S = S_list[0] * np.repeat(signs_sub_0, n).reshape(p, n)
#     return np.array([np.sign(np.sum(S * Si, axis=1)) for Si in S_list])

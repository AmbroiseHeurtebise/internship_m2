import numpy as np
import scipy
from tqdm import tqdm
from multiviewica_delay import _apply_delay_by_source, _apply_delay_one_source_or_sub
from apply_dilations_shifts import apply_dilations_shifts_1d, apply_dilations_shifts_3d_jax
from lbfgsb_loss_and_grad import smoothing_filter_3d


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


# initialize dilations and shifts with a gridsearch after permica
def grid_search_orders_dilations_shifts(S_list, max_dilation, max_shift, n_concat=1, nb_points_grid=10):
    m, p, _ = S_list.shape
    allowed_dilations = np.linspace(1/max_dilation, max_dilation, nb_points_grid)
    allowed_shifts = np.linspace(-max_shift, max_shift, nb_points_grid)
    grid_d, grid_s = np.meshgrid(allowed_dilations, allowed_shifts)
    grid_d = grid_d.ravel()
    grid_s = grid_s.ravel()
    S = S_list[0].copy()
    orders = np.zeros((m, p))
    orders[0] = np.arange(p)
    dilations = np.ones((m, p))
    shifts = np.zeros((m, p))
    for i in range(1, m):
        best_scores = np.zeros((p, p))
        best_dilation_shift = np.zeros((p, p))
        for j in range(p):
            s = S_list[i, j]
            scores = np.zeros((len(grid_d), p))
            for k, (dilation, shift) in enumerate(zip(grid_d, grid_s)):
                s_delayed = apply_dilations_shifts_1d(
                    s, dilation, shift, max_dilation, max_shift, False, n_concat)
                scores[k] = np.minimum(np.mean((S - s_delayed) ** 2, axis=1), np.mean((S + s_delayed) ** 2, axis=1))
            best_scores[j] = np.min(scores, axis=0)
            best_dilation_shift[j] = np.argmin(scores, axis=0)
        _, order = scipy.optimize.linear_sum_assignment(best_scores)
        orders[i] = order
        best_dilation_shift_idx = np.array([best_dilation_shift[l, order[l]] for l in range(p)]).astype(int)
        dilations[i] = np.array([grid_d[idx] for idx in best_dilation_shift_idx])
        shifts[i] = np.array([grid_s[idx] for idx in best_dilation_shift_idx])
    orders = orders.astype(int)
    return orders, dilations, shifts


def find_signs_sources(S_list):
    m, p, _ = S_list.shape
    S = S_list[0].copy()
    signs = np.ones((m, p))
    for i in range(1, m):
        signs[i] = 2 * np.argmin(
            np.vstack(
                [np.mean((S_list[i] + S) ** 2, axis=1),
                 np.mean((S_list[i] - S) ** 2, axis=1)]
                ), axis=0
            ) - 1
    signs = signs.astype(int)
    return signs


def permica_preprocessing(
    S_list_init, W_list_init, max_dilation, max_shift, n_concat, nb_points_grid=30, S_list_true=None, filter_length=20,
):
    print("\nPreprocess permica data...")
    m, p, n_total = S_list_init.shape
    # smoothing
    # S_list_init_smooth = smoothing_filter_3d(S_list_init, filter_length)
    S_list_init_smooth = S_list_init
    # find order, dilation and shift for each source of each subject
    orders_init, dilations_init, shifts_init = grid_search_orders_dilations_shifts(
        S_list_init_smooth, max_dilation=1+2*(max_dilation-1), max_shift=2*max_shift, n_concat=n_concat,
        nb_points_grid=nb_points_grid)
    S_list_init = apply_dilations_shifts_3d_jax(
        S_list_init, dilations_init, shifts_init, max_dilation=1+2*(max_dilation-1),
        max_shift=2*max_shift, shift_before_dilation=False, n_concat=n_concat)
    W_list_init = np.array([W_list_init[i][orders_init[i]] for i in range(m)])
    S_list_init = np.array([S_list_init[i][orders_init[i]] for i in range(m)])
    dilations_init = np.array([dilations_init[i][orders_init[i]] for i in range(m)])
    shifts_init = np.array([shifts_init[i][orders_init[i]] for i in range(m)])
    # find sign
    signs = find_signs_sources(S_list_init)
    W_list_init *= np.repeat(signs, p, axis=1).reshape(m, p, p)
    # S_list_init *= np.repeat(signs, n_total, axis=1).reshape(m, p, n_total)
    S_list_init *= signs[:, :, np.newaxis]
    # find the order that aligns S_list and S_list_init
    if S_list_true is not None:
        order_global = find_order(np.mean(S_list_true, axis=0), np.mean(S_list_init, axis=0))
        W_list_init = W_list_init[:, order_global, :]
        S_list_init = S_list_init[:, order_global, :]
        dilations_init = dilations_init[:, order_global]
        shifts_init = shifts_init[:, order_global]
        signs_0 = 2 * np.argmin(np.vstack(
            [np.mean((S_list_init[0] + S_list_true[0]) ** 2, axis=1),
             np.mean((S_list_init[0] - S_list_true[0]) ** 2, axis=1)]), axis=0) - 1
        signs_0 = signs_0.astype(int)
        W_list_init[0] *= signs_0[:, np.newaxis]
        S_list_init[0] *= signs_0[:, np.newaxis]
        signs = find_signs_sources(S_list_init)
        W_list_init *= np.repeat(signs, p, axis=1).reshape(m, p, p)
        S_list_init *= np.repeat(signs, n_total, axis=1).reshape(m, p, n_total)
    S_list_init = apply_dilations_shifts_3d_jax(
        S_list_init, 1/dilations_init, -shifts_init, max_dilation=1+2*(max_dilation-1),
        max_shift=2*max_shift, shift_before_dilation=True, n_concat=n_concat)
    print("Preprocessing done.")
    return S_list_init, W_list_init

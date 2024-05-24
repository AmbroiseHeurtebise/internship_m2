import numpy as np
import scipy
from apply_dilations_shifts import (
    apply_dilations_shifts_3d_no_argmin,
    apply_dilations_shifts_1d,
)


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
        best_dilation_shift_idx = np.array([best_dilation_shift[k, order[k]] for k in range(p)]).astype(int)
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
    W_list_permica,
    X_list,
    max_dilation=1.15,
    max_shift=0.05,
    n_concat=1,
    nb_points_grid=20,
    S_list_true=None,
    verbose=False,
):
    if verbose:
        print("\nPreprocess permica data...")
    S_list_permica = np.array([np.dot(W, X) for W, X in zip(W_list_permica, X_list)])
    m, p, n_total = S_list_permica.shape
    # find order, dilation and shift for each source of each subject
    orders_permica, dilations_permica, shifts_permica = grid_search_orders_dilations_shifts(
        S_list_permica, max_dilation=max_dilation, max_shift=max_shift, n_concat=n_concat,
        nb_points_grid=nb_points_grid)
    S_list_permica = apply_dilations_shifts_3d_no_argmin(
        S_list_permica, dilations=dilations_permica, shifts=shifts_permica, max_dilation=max_dilation,
        max_shift=max_shift, shift_before_dilation=False, n_concat=n_concat)
    W_list_permica = np.array([W_list_permica[i][orders_permica[i]] for i in range(m)])
    S_list_permica = np.array([S_list_permica[i][orders_permica[i]] for i in range(m)])
    dilations_permica = np.array([dilations_permica[i][orders_permica[i]] for i in range(m)])
    shifts_permica = np.array([shifts_permica[i][orders_permica[i]] for i in range(m)])
    # find sign
    signs = find_signs_sources(S_list_permica)
    W_list_permica *= np.repeat(signs, p, axis=1).reshape(m, p, p)
    S_list_permica *= signs[:, :, np.newaxis]
    # find the order that aligns S_list and S_list_init; only used for synthetic experiments
    if S_list_true is not None:
        order_global = find_order(np.mean(S_list_true, axis=0), np.mean(S_list_permica, axis=0))
        W_list_permica = W_list_permica[:, order_global, :]
        S_list_permica = S_list_permica[:, order_global, :]
        dilations_permica = dilations_permica[:, order_global]
        shifts_permica = shifts_permica[:, order_global]
        signs_0 = 2 * np.argmin(np.vstack(
            [np.mean((S_list_permica[0] + S_list_true[0]) ** 2, axis=1),
             np.mean((S_list_permica[0] - S_list_true[0]) ** 2, axis=1)]), axis=0) - 1
        signs_0 = signs_0.astype(int)
        W_list_permica[0] *= signs_0[:, np.newaxis]
        S_list_permica[0] *= signs_0[:, np.newaxis]
        signs = find_signs_sources(S_list_permica)
        W_list_permica *= np.repeat(signs, p, axis=1).reshape(m, p, p)
        S_list_permica *= np.repeat(signs, n_total, axis=1).reshape(m, p, n_total)
    S_list_permica = apply_dilations_shifts_3d_no_argmin(
        S_list_permica, 1/dilations_permica, -shifts_permica, max_dilation=max_dilation,
        max_shift=max_shift, shift_before_dilation=True, n_concat=n_concat)
    if verbose:
        print("Preprocessing done.")
    return S_list_permica, W_list_permica, dilations_permica, shifts_permica

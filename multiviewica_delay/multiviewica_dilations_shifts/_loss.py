import numpy as np
import jax.numpy as jnp
from ._apply_dilations_shifts import apply_dilations_shifts_3d


def penalization(dilations, shifts, max_dilation, max_shift):
    avg_dilations_minus_1 = jnp.mean(dilations, axis=0) - 1
    denominator = max_dilation * (avg_dilations_minus_1 >= 0) + 1 / max_dilation * (avg_dilations_minus_1 < 0) - 1
    pen_dilations = jnp.sum((avg_dilations_minus_1 / denominator) ** 2)
    pen_shifts = jnp.sum(jnp.mean(shifts / (max_shift), axis=0) ** 2)
    return pen_dilations + pen_shifts


def smoothing_filter_3d(S_list, filter_length):
    m, p, n = S_list.shape
    filter = np.ones(filter_length) / filter_length
    S_list_smooth = jnp.zeros((m, p, n))
    for i in range(m):
        S_list_smooth = S_list_smooth.at[i].set(
            jnp.array([jnp.convolve(s, filter, mode='same') for s in S_list[i]]))
    return S_list_smooth


def _logcosh(X):
    Y = jnp.abs(X)
    return Y + jnp.log1p(jnp.exp(-2 * Y))


def loss(
    W_dilations_shifts,
    X_list,
    dilation_scale,
    shift_scale,
    max_shift,
    max_dilation,
    noise_model,
    number_of_filters_envelop,
    filter_length_envelop,
    number_of_filters_squarenorm_f,
    filter_length_squarenorm_f,
    use_envelop_term,
    n_concat,
    penalization_scale,
    onset=0,
):
    m, p, _ = X_list.shape
    W_list = W_dilations_shifts[:m*p**2].reshape((m, p, p))
    dilations = W_dilations_shifts[m*p**2: m*p*(p+1)].reshape((m, p)) / dilation_scale
    shifts = W_dilations_shifts[m*p*(p+1):].reshape((m, p)) / shift_scale
    S_list = jnp.array([jnp.dot(W, X) for W, X in zip(W_list, X_list)])
    Y_list = apply_dilations_shifts_3d(
        S_list, dilations=dilations, shifts=shifts, max_dilation=max_dilation,
        max_shift=max_shift, shift_before_dilation=False, n_concat=n_concat, onset=onset)
    # shifts and dilations' penalization term
    loss = penalization_scale * penalization(dilations, shifts, max_dilation, max_shift)
    # envelop fitting term
    if use_envelop_term:
        Y_abs_smooth = jnp.abs(Y_list)
        for _ in range(number_of_filters_envelop):
            Y_abs_smooth = smoothing_filter_3d(Y_abs_smooth, filter_length_envelop)
        Y_abs_smooth_avg = jnp.mean(Y_abs_smooth, axis=0)
        loss += 1 / (2 * noise_model) * jnp.sum(
            jnp.array([jnp.mean((Y - Y_abs_smooth_avg) ** 2) for Y in Y_abs_smooth])) * p
    # fitting term
    for _ in range(number_of_filters_squarenorm_f):
        Y_list = smoothing_filter_3d(Y_list, filter_length_squarenorm_f)
    Y_avg = jnp.mean(Y_list, axis=0)
    loss += 1 / (2 * noise_model) * jnp.sum(
        jnp.array([jnp.mean((Y - Y_avg) ** 2) for Y in Y_list])) * p
    # function f
    loss += jnp.mean(_logcosh(Y_avg)) * p
    # log(det(W))
    for W in W_list:
        loss -= jnp.linalg.slogdet(W)[1]
    return loss

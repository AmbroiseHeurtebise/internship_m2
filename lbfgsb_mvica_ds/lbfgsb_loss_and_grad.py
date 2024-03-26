import numpy as np
import jax.numpy as jnp
from apply_dilations_shifts import (
    apply_dilations_shifts,
    apply_both_delays_3d_cyclic,
    apply_dilations_shifts_3d_jax,
    apply_dilations_shifts_4d_jax,
    apply_dilations_shifts_3d_no_argmin,
    apply_dilations_shifts_4d_no_argmin,
)


def penalization(dilations, shifts, max_dilation, max_shift):
    pen_dilations = jnp.sum(
        jnp.mean(2 * (dilations - 1) / (max_dilation - 1 / max_dilation), axis=0) ** 2)
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
    W_A_B,
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
    # apply_delays_function,
):
    m, p, _ = X_list.shape
    W_list = W_A_B[:m*p**2].reshape((m, p, p))
    A = W_A_B[m*p**2: m*p*(p+1)].reshape((m, p)) / dilation_scale
    B = W_A_B[m*p*(p+1):].reshape((m, p)) / shift_scale
    S_list = jnp.array([jnp.dot(W, X) for W, X in zip(W_list, X_list)])

    # ########################################## temporary block ###########################################
    which_method = 4
    if which_method == 0:  # for loops
        Y_list = apply_both_delays_3d_cyclic(
            S_list, dilations=A, shifts=B, max_shift=max_shift, max_dilation=max_dilation,
            shift_before_dilation=False, n_concat=n_concat)
    elif which_method == 1:  # vectorize with many if else conditions
        S_list_4d = jnp.moveaxis(jnp.array(jnp.split(S_list, n_concat, axis=-1)), source=0, destination=2)
        Y_list_4d = apply_dilations_shifts(
            S_list_4d, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift,
            shift_before_dilation=False)
        Y_list = Y_list_4d.reshape((m, p, -1))
    elif which_method == 2:  # vectorize without any if else condition
        S_list_4d = jnp.moveaxis(jnp.array(jnp.split(S_list, n_concat, axis=-1)), source=0, destination=2)
        Y_list_4d = apply_dilations_shifts_4d_jax(
            S_list_4d, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift)
        Y_list = Y_list_4d.reshape((m, p, -1))
    elif which_method == 3:  # vectorize in 3d instead of 4d
        Y_list = apply_dilations_shifts_3d_jax(
            S_list, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift, n_concat=n_concat)
    elif which_method == 4:
        Y_list = apply_dilations_shifts_3d_no_argmin(
            S_list, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift,
            shift_before_dilation=False, n_concat=n_concat)
    elif which_method == 5:
        S_list_4d = jnp.moveaxis(jnp.array(jnp.split(S_list, n_concat, axis=-1)), source=0, destination=2)
        Y_list_4d = apply_dilations_shifts_4d_no_argmin(
            S_list_4d, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift,
            shift_before_dilation=False)
        Y_list = Y_list_4d.reshape((m, p, -1))
    else:
        raise ValueError("which_number should be 0, 1, 2 or 3.")
    # Y_list = apply_delays_function(
    #     S_list, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift,
    #     shift_before_dilation=False, n_concat=n_concat)
    # ######################################### end of the block ###########################################

    # shifts and dilations' penalization term
    penalization_factor = 1 / (2 * noise_model) * p * m * penalization_scale
    loss = penalization_factor * penalization(A, B, max_dilation, max_shift)
    # envelop fitting term
    if use_envelop_term:
        Y_abs_smooth = jnp.abs(Y_list)
        # Y_abs_smooth = Y_list ** 2  # XXX now it is differentiable; just a test
        for _ in range(number_of_filters_envelop):
            # Y_abs_smooth = smoothing_filter_3d_envelop(Y_abs_smooth)
            Y_abs_smooth = smoothing_filter_3d(Y_abs_smooth, filter_length_envelop)
        Y_abs_smooth_avg = jnp.mean(Y_abs_smooth, axis=0)
        # Y_abs_smooth_avg = Y_abs_smooth[m // 2]
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

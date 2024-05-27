import numpy as np
import jax.numpy as jnp
from .apply_dilations_shifts import apply_dilations_shifts_3d_no_argmin


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
):
    m, p, _ = X_list.shape
    W_list = W_A_B[:m*p**2].reshape((m, p, p))
    A = W_A_B[m*p**2: m*p*(p+1)].reshape((m, p)) / dilation_scale
    B = W_A_B[m*p*(p+1):].reshape((m, p)) / shift_scale
    S_list = jnp.array([jnp.dot(W, X) for W, X in zip(W_list, X_list)])
    Y_list = apply_dilations_shifts_3d_no_argmin(
        S_list, dilations=A, shifts=B, max_dilation=max_dilation, max_shift=max_shift,
        shift_before_dilation=False, n_concat=n_concat)
    # shifts and dilations' penalization term
    loss = penalization_scale * penalization(A, B, max_dilation, max_shift)
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

import numpy as np
import jax.numpy as jnp
from jax import vmap


def interp_mapped(*args):
    return vmap(vmap(jnp.interp, (0, None, 0)), (1, None, None), out_axes=1)(*args)


def apply_dilations_shifts_3d(
    S, dilations, shifts, max_dilation=1., max_shift=0., shift_before_dilation=True, n_concat=1,
):
    """ Apply dilations and shifts to sources.
    It is suited for jax computations.

    Args:
        S (nd.array):
            Sources of shape (m, p, n_concat*n).
        dilations (nd.array):
            There is one dilation factor for each source. So, dilations' shape is (m, p).
        shifts (nd.array):
            There is one shift factor for each source. So, shifts' shape is (m, p).
        max_dilation (float, optional):
            Maximum dilation factor allowed. Dilations are in [1/max_dilation, max_dilation].
            Defaults to 1.
        max_shift (float, optional):
            Maximum shift factor allowed. Shifts are in [-max_shift, max_shift].
            Defaults to 0.
        shift_before_dilation (bool, optional)
            Decide if we apply shift before or after dilation.
            If True: s(t) <- s(rho (t + tau)).
            If False: s(t) <- s(rho t + tau).
            Default is True.
        n_concat (int, optional)
            Number of concatenations, i.e. number of epochs in the context of M/EEG data.
            Default to 1.

    Returns:
        S_ds (nd.array):
            Sources after applying dilations and shifts.
            Their shape is the same as S, i.e. (m, p, n_concat*n).
    """
    m, p, n_total = S.shape
    n = n_total // n_concat
    S_4d = jnp.moveaxis(jnp.array(jnp.split(S, n_concat, axis=-1)), source=0, destination=2)
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    t_extended = jnp.arange(n+2*max_delay_samples) - max_delay_samples
    t = jnp.arange(n)
    T = jnp.array([t] * m * p * n_concat).reshape(S_4d.shape)
    dilations_newaxis = dilations[:, :, jnp.newaxis, jnp.newaxis]
    shifts_newaxis = shifts[:, :, jnp.newaxis, jnp.newaxis] * n
    if shift_before_dilation:
        T_ds = (T - shifts_newaxis) * dilations_newaxis
    else:
        T_ds = T * dilations_newaxis - shifts_newaxis
    T_ds_clipped = jnp.clip(T_ds, -max_delay_samples+1, n+max_delay_samples-2)
    T_ref = jnp.rint(T_ds).astype(int)
    ind = T_ref + max_delay_samples
    # jnp.copysign allows to avoid the case sign==0
    signs = jnp.copysign(1, jnp.sign(T_ds_clipped - T_ref)).astype(int)
    S_extended = jnp.concatenate(
        [S_4d[:, :, :, n-max_delay_samples:], S_4d, S_4d[:, :, :, :max_delay_samples]], axis=-1)
    S_extended_ind = jnp.take_along_axis(S_extended, ind, axis=-1)
    t_extended_ind = t_extended[ind]
    slopes = (jnp.take_along_axis(S_extended, ind+signs, axis=-1) - S_extended_ind) / (
        t_extended[ind + signs] - t_extended_ind)
    intercepts = S_extended_ind - slopes * t_extended_ind
    S_ds = slopes * T_ds + intercepts
    S_ds = S_ds.reshape((m, p, -1))
    return S_ds


def apply_dilations_shifts_1d(
    s, dilation, shift, max_dilation=1., max_shift=0., shift_before_dilation=True, n_concat=1
):
    assert len(s) % n_concat == 0
    S = np.array(np.split(s, n_concat))
    n = S.shape[1]
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    t_extended = jnp.arange(n+2*max_delay_samples) - max_delay_samples
    t = jnp.arange(n)
    shift *= n
    if shift_before_dilation:
        t_ds = (t - shift) * dilation
    else:
        t_ds = t * dilation - shift
    t_ds_clipped = np.clip(t_ds, -max_delay_samples+1, n+max_delay_samples-2)
    t_ref = t_ds_clipped.astype(int)
    ind = (t_ref + max_delay_samples)[np.newaxis, :]
    signs = np.copysign(1, np.sign(t_ds_clipped - t_ref)).astype(int)
    S_extended = np.concatenate([S[:, n-max_delay_samples:], S, S[:, :max_delay_samples]], axis=-1)
    S_extended_ind = np.take_along_axis(S_extended, ind, axis=1)
    t_extended_ind = t_extended[ind]
    slopes = (np.take_along_axis(S_extended, ind+signs, axis=1) - S_extended_ind) / (
        t_extended[ind + signs] - t_extended_ind)
    intercepts = S_extended_ind - slopes * t_extended_ind
    S_ds = slopes * t_ds + intercepts
    s_ds = S_ds.ravel()
    return s_ds

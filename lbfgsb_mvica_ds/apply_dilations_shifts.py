import numpy as np
import jax.numpy as jnp
from jax import vmap


def interp_mapped(*args):
    return vmap(vmap(jnp.interp, (0, None, 0)), (1, None, None), out_axes=1)(*args)


def apply_dilations_shifts(S, dilations, shifts, max_dilation=1., max_shift=0., shift_before_dilation=True):
    """Apply dilations and shifts to sources.

    Args:
        S (nd.array):
            The shape of sources is (p, n), (m, p, n), or (m, p, n_concat, n).
        dilations (nd.array):
            There is one dilation factor for each source. So, dilations' shape is (p,) or (m, p).
        shifts (nd.array):
            There is one shift factor for each source. So, shifts' shape is (p,) or (m, p).
        max_dilation (float, optional):
            Maximum dilation factor allowed. Dilations are in [1/max_dilation, max_dilation].
            Defaults to 1.
        max_shift (float, optional):
            Maximum shift factor allowed. Shifts are in [-max_shift, max_shift].
            Defaults to 0.
        shift_before_dilation (bool, optional):
            If True: apply shift before dilation.
            Else: apply dilation before shift.
            Defaults to True.

    Returns:
        S_ds (nd.array):
            Sources after applying dilations and shifts.
            Their shape is the same as S, i.e. (p, n), (m, p, n), or (m, p, n_concat, n).
    """
    # assert (1 / max_dilation <= dilations).all() and (dilations <= max_dilation).all()
    # assert (-max_shift <= shifts).all() and (shifts <= max_shift).all()
    if np.ndim(S) == 2:
        p, n = S.shape
        assert dilations.shape == shifts.shape == (p,)
        m_p_nconcat = p
    elif np.ndim(S) == 3:
        m, p, n = S.shape
        assert dilations.shape == shifts.shape == (m, p)
        m_p_nconcat = m * p
    elif np.ndim(S) == 4:
        m, p, n_concat, n = S.shape
        assert dilations.shape == shifts.shape == (m, p)
        m_p_nconcat = m * p * n_concat
    else:
        raise ValueError("The number of dimensions of S should be 2, 3 or 4.")
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    t_extended = np.linspace(-max_delay_time, 1+max_delay_time, n+2*max_delay_samples)
    t = np.linspace(0, 1, n)
    T = np.array([t] * m_p_nconcat).reshape(S.shape)
    if np.ndim(S) == 2 or np.ndim(S) == 3:
        dilations_newaxis = jnp.expand_dims(dilations, axis=-1)
        shifts_newaxis = jnp.expand_dims(shifts, axis=-1)
    else:
        dilations_newaxis = jnp.expand_dims(dilations, axis=(-1, -2))
        shifts_newaxis = jnp.expand_dims(shifts, axis=(-1, -2))
    if shift_before_dilation:
        T_ds = (T - shifts_newaxis) * dilations_newaxis
    else:
        T_ds = T * dilations_newaxis - shifts_newaxis
    T_ds_ravel = T_ds.reshape((-1, n))
    if np.ndim(S) == 2:
        S_extended = jnp.concatenate([S[:, n-max_delay_samples:], S, S[:, :max_delay_samples]], axis=-1)
    elif np.ndim(S) == 3:
        S_extended = jnp.concatenate([S[:, :, n-max_delay_samples:], S, S[:, :, :max_delay_samples]], axis=-1)
    else:
        S_extended = jnp.concatenate([S[:, :, :, n-max_delay_samples:], S, S[:, :, :, :max_delay_samples]], axis=-1)
    S_extended_ravel = S_extended.reshape((-1, n+2*max_delay_samples))
    S_ds = interp_mapped(T_ds_ravel, t_extended, S_extended_ravel)
    S_ds = S_ds.reshape(S.shape)
    return S_ds


def apply_both_delays_2d_cyclic(
    S,
    a,
    b,
    max_shift=0.,
    max_dilation=1.,
    shift_before_dilation=True,
    n_concat=1,
):
    p, n_total = S.shape
    n = n_total // n_concat
    max_delay = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay * n).astype("int")
    t_extended = jnp.linspace(-max_delay, 1+max_delay, n+2*max_delay_samples)
    t = jnp.linspace(0, 1, n)
    T = jnp.array([t] * p)
    if shift_before_dilation:
        T_ab = ((T.T - b) * a).T
    else:
        T_ab = (T.T * a - b).T
    S_split = jnp.array(jnp.hsplit(S, n_concat))
    S_ab = jnp.zeros_like(S)
    for i in range(n_concat):
        S_i = S_split[i]
        S_extended = jnp.concatenate([S_i[:, n-max_delay_samples:], S_i, S_i[:, :max_delay_samples]], axis=1)
        S_ab = S_ab.at[:, i*n: (i+1)*n].set(
            jnp.array([jnp.interp(x=T_ab[i], xp=t_extended, fp=S_extended[i], left=0, right=0)
                       for i in range(p)]))
    return S_ab


def apply_both_delays_3d_cyclic(
    S_list,
    dilations,
    shifts,
    max_dilation=1.,
    max_shift=0.,
    shift_before_dilation=True,
    n_concat=1,
):
    Y_list = jnp.array(
        [apply_both_delays_2d_cyclic(
            S_list[i], a=dilations[i], b=shifts[i], max_dilation=max_dilation, max_shift=max_shift,
            shift_before_dilation=shift_before_dilation, n_concat=n_concat)
         for i in range(len(S_list))])
    return Y_list


def apply_dilations_shifts_4d_jax(S, dilations, shifts, max_dilation=1., max_shift=0.):
    """ Apply dilations and shifts to sources.
    This function corresponds to apply_dilations_shifts(),
    in the case where S has 4 dimensions and shift_before_dilation=False.
    It is suited for jax computations.

    Args:
        S (nd.array):
            Sources of shape (m, p, n_concat, n).
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

    Returns:
        S_ds (nd.array):
            Sources after applying dilations and shifts.
            Their shape is the same as S, i.e. (m, p, n_concat, n).
    """
    m, p, n_concat, n = S.shape
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    t_extended = jnp.linspace(-max_delay_time, 1+max_delay_time, n+2*max_delay_samples)
    t = jnp.linspace(0, 1, n)
    T = jnp.array([t] * m * p * n_concat).reshape(S.shape)
    dilations_newaxis = dilations[:, :, jnp.newaxis, jnp.newaxis]
    shifts_newaxis = shifts[:, :, jnp.newaxis, jnp.newaxis]
    T_ds = T * dilations_newaxis - shifts_newaxis  # corresponds to shift_before_dilation=False
    T_ds_ravel = T_ds.reshape((-1, n))
    S_extended = jnp.concatenate([S[:, :, :, n-max_delay_samples:], S, S[:, :, :, :max_delay_samples]], axis=-1)
    S_extended_ravel = S_extended.reshape((-1, n+2*max_delay_samples))
    S_ds = interp_mapped(T_ds_ravel, t_extended, S_extended_ravel)
    S_ds = S_ds.reshape(S.shape)
    return S_ds


def apply_dilations_shifts_3d_jax(
    S, dilations, shifts, max_dilation=1., max_shift=0., shift_before_dilation=True, n_concat=1,
):
    m, p, n_total = S.shape
    n = n_total // n_concat
    S_4d = jnp.moveaxis(jnp.array(jnp.split(S, n_concat, axis=-1)), source=0, destination=2)
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    t_extended = jnp.linspace(-max_delay_time, 1+max_delay_time, n+2*max_delay_samples)
    t = jnp.linspace(0, 1, n)
    T = jnp.array([t] * m * p * n_concat).reshape((m, p, n_concat, n))
    dilations_newaxis = dilations[:, :, jnp.newaxis, jnp.newaxis]
    shifts_newaxis = shifts[:, :, jnp.newaxis, jnp.newaxis]
    if shift_before_dilation:
        T_ds = (T - shifts_newaxis) * dilations_newaxis
    else:
        T_ds = T * dilations_newaxis - shifts_newaxis
    T_ds_ravel = T_ds.reshape((-1, n))
    S_extended = jnp.concatenate(
        [S_4d[:, :, :, n-max_delay_samples:], S_4d, S_4d[:, :, :, :max_delay_samples]], axis=-1)
    S_extended_ravel = S_extended.reshape((-1, n+2*max_delay_samples))
    S_ds = interp_mapped(T_ds_ravel, t_extended, S_extended_ravel)
    S_ds = S_ds.reshape(S.shape)
    return S_ds


def apply_dilations_shifts_3d_no_argmin(
    S, dilations, shifts, max_dilation=1., max_shift=0., shift_before_dilation=True, n_concat=1,
):
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
    # ind = jnp.rint(T_ds).astype(int) + max_delay_samples
    # ind = jnp.clip(ind, 1, n+2*max_delay_samples-2)
    # T_ref = t_extended[ind]
    # T_ref = jnp.clip(jnp.rint(T_ds).astype(int), -max_delay_samples+1, n+max_delay_samples-2)
    # # ind = jnp.clip(T_ref + max_delay_samples, 1, n+2*max_delay_samples-2)
    # ind = T_ref + max_delay_samples
    # T_ds_clipped = jnp.clip(T_ds, -max_delay_samples+1, n+max_delay_samples-2)
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


def apply_dilations_shifts_4d_no_argmin(
    S, dilations, shifts, max_dilation=1., max_shift=0., shift_before_dilation=True,
):
    m, p, n_concat, n = S.shape
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    t_extended = jnp.arange(n+2*max_delay_samples) - max_delay_samples
    t = jnp.arange(n)
    T = jnp.array([t] * m * p * n_concat).reshape(S.shape)
    dilations_newaxis = dilations[:, :, jnp.newaxis, jnp.newaxis]
    shifts_newaxis = shifts[:, :, jnp.newaxis, jnp.newaxis] * n
    if shift_before_dilation:
        T_ds = (T - shifts_newaxis) * dilations_newaxis
    else:
        T_ds = T * dilations_newaxis - shifts_newaxis
    # ind = jnp.rint(T_ds).astype(int) + max_delay_samples
    # ind = jnp.clip(ind, 1, n+2*max_delay_samples-2)
    # T_ref = t_extended[ind]
    T_ref = jnp.clip(jnp.rint(T_ds).astype(int), -max_delay_samples+1, n+max_delay_samples-2)
    ind = jnp.clip(T_ref + max_delay_samples, 1, n+2*max_delay_samples-2)
    T_ds_clipped = jnp.clip(T_ds, -max_delay_samples+1, n+max_delay_samples-2)
    # jnp.copysign allows to avoid the case sign==0
    signs = jnp.copysign(1, jnp.sign(T_ds_clipped - T_ref)).astype(int)
    S_extended = jnp.concatenate([S[:, :, :, n-max_delay_samples:], S, S[:, :, :, :max_delay_samples]], axis=-1)
    S_extended_ind = jnp.take_along_axis(S_extended, ind, axis=-1)
    t_extended_ind = t_extended[ind]
    slopes = (jnp.take_along_axis(S_extended, ind+signs, axis=-1) - S_extended_ind) / (
        t_extended[ind + signs] - t_extended_ind)
    intercepts = S_extended_ind - slopes * t_extended_ind
    S_ds = slopes * T_ds + intercepts
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

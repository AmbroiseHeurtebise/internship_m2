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
    A,
    B,
    max_shift=0.,
    max_dilation=1.,
    shift_before_dilation=True,
    n_concat=1,
):
    Y_list = jnp.array(
        [apply_both_delays_2d_cyclic(
            S_list[i], a=A[i], b=B[i], max_shift=max_shift, max_dilation=max_dilation,
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


def apply_dilations_shifts_3d_jax(S, dilations, shifts, max_dilation=1., max_shift=0., n_concat=1):
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
    T_ds = T * dilations_newaxis - shifts_newaxis  # corresponds to shift_before_dilation=False
    T_ds_ravel = T_ds.reshape((-1, n))
    S_extended = jnp.concatenate(
        [S_4d[:, :, :, n-max_delay_samples:], S_4d, S_4d[:, :, :, :max_delay_samples]], axis=-1)
    S_extended_ravel = S_extended.reshape((-1, n+2*max_delay_samples))
    S_ds = interp_mapped(T_ds_ravel, t_extended, S_extended_ravel)
    S_ds = S_ds.reshape(S.shape)
    return S_ds

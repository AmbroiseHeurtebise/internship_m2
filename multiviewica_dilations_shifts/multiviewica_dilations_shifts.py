import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b
from time import time
from multiviewica_delay import permica

from loss import loss
from other_functions import Memory_callback, compute_dilation_shift_scales
from apply_dilations_shifts import apply_dilations_shifts_3d_no_argmin
from permica_preprocessing import permica_preprocessing


jax.config.update('jax_enable_x64', True)


def mvica_ds(
    X_list,
    n_concat,
    max_dilation,
    max_shift,
    W_scale,
    dilation_scale_per_source,
    bounds_factor,
    verbose,
    random_state,
    noise_model,
    number_of_filters_envelop,
    filter_length_envelop,
    number_of_filters_squarenorm_f,
    filter_length_squarenorm_f,
    use_envelop_term,
    penalization_scale,
    return_all_iterations,
    nb_points_grid_init=20,
):
    m, p, n_total = X_list.shape
    n = n_total // n_concat

    # extend max_dilation and max_shift, only useful for synthetic experiments
    max_dilation_2 = 1 + bounds_factor * (max_dilation - 1)
    max_shift_2 = bounds_factor * max_shift

    # initialize W_list, S_list, dilations and shifts with permica
    max_delay = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay * n).astype("int")
    _, W_list_init, _, _ = permica(
        X_list, max_iter=100, random_state=random_state, tol=1e-9,
        optim_delays=True, max_delay=max_delay_samples)
    S_list_init = np.array([np.dot(W, X) for W, X in zip(W_list_init, X_list)])
    S_list_init, W_list_init, dilations_init, shifts_init = permica_preprocessing(
        S_list_init, W_list_init, max_dilation, max_shift, n_concat,
        nb_points_grid=nb_points_grid_init, verbose=verbose)

    # shift and dilation scales
    dilation_scale, shift_scale = compute_dilation_shift_scales(
        max_dilation, max_shift, W_scale, dilation_scale_per_source,
        S_list_init, n_concat)

    # initialize A and B
    dilations_init_c = dilations_init - np.mean(dilations_init, axis=0) + 1
    shifts_init_c = shifts_init - np.mean(shifts_init, axis=0)
    A_init = dilations_init_c * dilation_scale
    B_init = shifts_init_c * shift_scale
    W_A_B_init = jnp.concatenate([jnp.ravel(W_list_init), jnp.ravel(A_init), jnp.ravel(B_init)])

    # arguments for L-BFGS
    args = (
        W_A_B_init,
        X_list,
        dilation_scale,
        shift_scale,
        max_shift_2,
        max_dilation_2,
        noise_model,
        number_of_filters_envelop,
        filter_length_envelop,
        number_of_filters_squarenorm_f,
        filter_length_squarenorm_f,
        use_envelop_term,
        n_concat,
        penalization_scale)

    # jit
    val_and_grad = jax.jit(jax.value_and_grad(loss), static_argnums=tuple(np.arange(3, 14)))

    def wrapper_loss_and_grad(*args):
        val, grad = val_and_grad(*args)
        return val, np.array(grad)

    if verbose:
        print("Jit...")
        start = time()
    wrapper_loss_and_grad(*args)
    if verbose:
        print(f"Jit time : {time() - start}")

    # bounds
    bounds_W = [(-jnp.inf, jnp.inf)] * (m * p**2)
    if dilation_scale_per_source:
        bounds_A = [
            (1 / max_dilation_2 * dilation_scale.ravel()[i], max_dilation_2 * dilation_scale.ravel()[i])
            for i in range(m * p)]
    else:
        bounds_A = [
            (1 / max_dilation_2 * dilation_scale, max_dilation_2 * dilation_scale)] * (m * p)
    bounds_B = [(-max_shift_2 * shift_scale, max_shift_2 * shift_scale)] * (m * p)
    bounds_W_A_B = jnp.array(bounds_W + bounds_A + bounds_B)

    # LBFGSB
    callback = Memory_callback(m, p, dilation_scale, shift_scale)
    if verbose:
        print("LBFGSB...")
        start = time()
    fmin_l_bfgs_b(
        func=wrapper_loss_and_grad,
        x0=args[0],
        args=args[1:],
        bounds=bounds_W_A_B,
        disp=True,
        factr=1e3,
        pgtol=1e-5,
        maxiter=3000,
        callback=callback,
    )
    if verbose:
        print(f"LBFGSB time : {time() - start}")

    # get parameters of the last iteration
    memory_W = np.array(callback.memory_W)
    memory_A = np.array(callback.memory_A)
    memory_B = np.array(callback.memory_B)
    W_lbfgs = memory_W[-1]
    A_lbfgs = 1 / memory_A[-1]
    B_lbfgs = -memory_B[-1]

    # reconstruct sources
    S_list_lbfgsb = jnp.array([jnp.dot(W, X) for W, X in zip(W_lbfgs, X_list)])
    Y_list_lbfgsb = apply_dilations_shifts_3d_no_argmin(
        S_list_lbfgsb, dilations=1/A_lbfgs, shifts=-B_lbfgs, max_shift=max_shift_2,
        max_dilation=max_dilation_2, shift_before_dilation=False)

    if return_all_iterations:
        return W_lbfgs, A_lbfgs, B_lbfgs, Y_list_lbfgsb, callback, W_list_init, dilations_init, shifts_init
    return W_lbfgs, A_lbfgs, B_lbfgs, Y_list_lbfgsb

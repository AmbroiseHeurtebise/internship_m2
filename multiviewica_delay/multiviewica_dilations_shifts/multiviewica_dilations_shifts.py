import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b
from time import time

from multiviewica_delay.multiviewica_shifts._multiviewica_shifts import permica
from .loss import loss
from .other_functions import Memory_callback, compute_dilation_shift_scales
from .apply_dilations_shifts import apply_dilations_shifts_3d_no_argmin
from .permica_preprocessing import permica_preprocessing


jax.config.update('jax_enable_x64', True)


def mvica_ds(
    X_list,
    n_concat,
    max_dilation,
    max_shift,
    W_scale,
    dilation_scale_per_source,
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
    S_list_true=None,
):
    m, p, n_total = X_list.shape

    # initialize W_list, S_list, dilations and shifts with permica
    _, W_list_permica, _, _ = permica(
        X_list, max_iter=1000, random_state=random_state, tol=1e-9,
        optim_delays=False)
    S_list_permica, W_list_permica, dilations_permica, shifts_permica = permica_preprocessing(
        W_list_permica=W_list_permica, X_list=X_list, max_dilation=max_dilation, max_shift=max_shift,
        n_concat=n_concat, nb_points_grid=nb_points_grid_init, S_list_true=S_list_true, verbose=verbose)

    # shift and dilation scales
    dilation_scale, shift_scale = compute_dilation_shift_scales(
        max_dilation=max_dilation, max_shift=max_shift, W_scale=W_scale,
        dilation_scale_per_source=dilation_scale_per_source, S_list=S_list_permica, n_concat=n_concat)

    # initialize W, dilations and shifts
    dilations_permica_c = dilations_permica - np.mean(dilations_permica, axis=0) + 1
    shifts_permica_c = shifts_permica - np.mean(shifts_permica, axis=0)
    dilations_init = dilations_permica_c * dilation_scale
    shifts_init = shifts_permica_c * shift_scale
    W_dilations_shifts_init = jnp.concatenate(
        [jnp.ravel(W_list_permica), jnp.ravel(dilations_init), jnp.ravel(shifts_init)])

    # arguments for L-BFGS
    args = (
        W_dilations_shifts_init,
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
        bounds_dilations = [
            (1 / max_dilation * dilation_scale.ravel()[i], max_dilation * dilation_scale.ravel()[i])
            for i in range(m * p)]
    else:
        bounds_dilations = [
            (1 / max_dilation * dilation_scale, max_dilation * dilation_scale)] * (m * p)
    bounds_shifts = [(-max_shift * shift_scale, max_shift * shift_scale)] * (m * p)
    bounds_W_dilations_shifts = jnp.array(bounds_W + bounds_dilations + bounds_shifts)

    # LBFGSB
    callback = Memory_callback(m, p, dilation_scale, shift_scale)
    if verbose:
        print("LBFGSB...")
        start = time()
    fmin_l_bfgs_b(
        func=wrapper_loss_and_grad,
        x0=args[0],
        args=args[1:],
        bounds=bounds_W_dilations_shifts,
        disp=True,
        factr=1e3,
        pgtol=1e-5,
        maxiter=3000,
        callback=callback,
    )
    if verbose:
        print(f"LBFGSB time : {time() - start}")

    # get parameters of the last iteration
    W_lbfgsb = np.array(callback.memory_W)[-1]
    dilations_lbfgsb = np.array(callback.memory_dilations)[-1]
    shifts_lbfgsb = np.array(callback.memory_shifts)[-1]

    # reconstruct sources
    S_list_lbfgsb = jnp.array([jnp.dot(W, X) for W, X in zip(W_lbfgsb, X_list)])
    Y_list_lbfgsb = apply_dilations_shifts_3d_no_argmin(
        S_list_lbfgsb, dilations=dilations_lbfgsb, shifts=shifts_lbfgsb, max_shift=max_shift,
        max_dilation=max_dilation, shift_before_dilation=False, n_concat=n_concat)

    if return_all_iterations:
        return (W_lbfgsb, dilations_lbfgsb, shifts_lbfgsb, Y_list_lbfgsb, callback, W_list_permica, dilations_permica,
                shifts_permica)
    return W_lbfgsb, dilations_lbfgsb, shifts_lbfgsb, Y_list_lbfgsb

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b
from time import time

from multiviewica_delay.multiviewica_shifts._reduce_data import reduce_data
from multiviewica_delay.multiviewica_shifts._multiviewica_shifts import permica
from ._loss import loss
from ._other_functions import Memory_callback, compute_dilation_shift_scales
from ._apply_dilations_shifts import apply_dilations_shifts_3d
from ._permica_processing import permica_processing


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
    n_components=None,
    dimension_reduction="pca",
):
    # dimensionality reduction
    P_list, X_list = reduce_data(
        X_list, n_components=n_components, dimension_reduction=dimension_reduction
    )
    m, p, n_total = X_list.shape

    # initialize W_list, S_list, dilations and shifts with permica
    _, W_list_permica, _, _ = permica(
        X_list, max_iter=1000, random_state=random_state, tol=1e-9,
        optim_delays=False)
    _, W_list_permica, dilations_permica, shifts_permica, S_avg_permica = permica_processing(
        W_list_permica=W_list_permica, X_list=X_list, max_dilation=max_dilation, max_shift=max_shift,
        n_concat=n_concat, nb_points_grid=nb_points_grid_init, S_list_true=S_list_true, verbose=verbose)

    # shift and dilation scales
    dilation_scale, shift_scale = compute_dilation_shift_scales(
        max_dilation=max_dilation, max_shift=max_shift, W_scale=W_scale,
        dilation_scale_per_source=dilation_scale_per_source, S_avg=S_avg_permica,
        n_concat=n_concat, m=m)

    # initialize W, dilations and shifts
    dilations_permica_c = dilations_permica - np.mean(dilations_permica, axis=0) + 1
    shifts_permica_c = shifts_permica - np.mean(shifts_permica, axis=0)
    dilations_init = dilations_permica_c * dilation_scale
    shifts_init = shifts_permica_c * shift_scale
    W_dilations_shifts_init = jnp.concatenate(
        [jnp.ravel(W_list_permica), jnp.ravel(dilations_init), jnp.ravel(shifts_init)])

    # arguments for L-BFGS
    kwargs = {
        "X_list": X_list,
        "dilation_scale": dilation_scale,
        "shift_scale": shift_scale,
        "max_shift": max_shift,
        "max_dilation": max_dilation,
        "noise_model": noise_model,
        "number_of_filters_envelop": number_of_filters_envelop,
        "filter_length_envelop": filter_length_envelop,
        "number_of_filters_squarenorm_f": number_of_filters_squarenorm_f,
        "filter_length_squarenorm_f": filter_length_squarenorm_f,
        "use_envelop_term": use_envelop_term,
        "n_concat": n_concat,
        "penalization_scale": penalization_scale,
    }

    # jit
    val_and_grad = jax.jit(
        jax.value_and_grad(loss),
        static_argnames=("shift_scale", "max_shift", "max_dilation", "noise_model",
                         "number_of_filters_envelop", "filter_length_envelop",
                         "number_of_filters_squarenorm_f", "filter_length_squarenorm_f",
                         "use_envelop_term", "n_concat", "penalization_scale"))

    def wrapper_loss_and_grad(W_dilations_shifts, kwargs):
        val, grad = val_and_grad(W_dilations_shifts, **kwargs)
        return val, np.array(grad)

    if verbose:
        print("Jit...")
        start = time()
    wrapper_loss_and_grad(W_dilations_shifts_init, kwargs)
    if verbose:
        print(f"Jit time : {time() - start}")

    # bounds
    bounds_W = [(-jnp.inf, jnp.inf)] * (m * p**2)
    if (not dilation_scale_per_source) or (max_dilation == 1):
        bounds_dilations = [
            (1 / max_dilation * dilation_scale, max_dilation * dilation_scale)] * (m * p)
    else:
        bounds_dilations = [
            (1 / max_dilation * dilation_scale.ravel()[i], max_dilation * dilation_scale.ravel()[i])
            for i in range(m * p)]
    bounds_shifts = [(-max_shift * shift_scale, max_shift * shift_scale)] * (m * p)
    bounds_W_dilations_shifts = jnp.array(bounds_W + bounds_dilations + bounds_shifts)

    # LBFGSB
    callback = Memory_callback(m, p, dilation_scale, shift_scale)
    if verbose:
        print("LBFGSB...")
        start = time()
    fmin_l_bfgs_b(
        func=wrapper_loss_and_grad,
        x0=W_dilations_shifts_init,
        args=(kwargs,),
        bounds=bounds_W_dilations_shifts,
        disp=verbose,
        factr=1e5,
        pgtol=1e-8,
        maxiter=3000,
        callback=callback,
    )
    if verbose:
        print(f"LBFGSB time : {time() - start}")

    # raise error in the case where L-BFGS-B didn't run
    if len(callback.memory_W) == 0:
        raise ValueError(
            "The algorithm immediately stopped before the first iteration. "
            "Maybe you used W_scale=0, max_dilation=1, max_shift=0, a too high pgtol, or a too low factr.")

    # get parameters of the last iteration
    W_lbfgsb = np.array(callback.memory_W)[-1]
    dilations_lbfgsb = np.array(callback.memory_dilations)[-1]
    shifts_lbfgsb = np.array(callback.memory_shifts)[-1]

    # reconstruct sources
    S_list_lbfgsb = jnp.array([jnp.dot(W, X) for W, X in zip(W_lbfgsb, X_list)])
    Y_list_lbfgsb = apply_dilations_shifts_3d(
        S_list_lbfgsb, dilations=dilations_lbfgsb, shifts=shifts_lbfgsb, max_shift=max_shift,
        max_dilation=max_dilation, shift_before_dilation=False, n_concat=n_concat)

    if return_all_iterations:
        return (W_lbfgsb, dilations_lbfgsb, shifts_lbfgsb, Y_list_lbfgsb, callback, W_list_permica, dilations_permica,
                shifts_permica)
    return W_lbfgsb, dilations_lbfgsb, shifts_lbfgsb, Y_list_lbfgsb

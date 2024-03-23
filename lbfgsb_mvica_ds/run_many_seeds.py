import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from time import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
from itertools import product

from picard import amari_distance

from multiviewica_delay import permica, multiviewica_delay
from generate_data import generate_data, generate_new_data
from permica_preprocessing import permica_preprocessing
from other_functions import compute_lambda
from lbfgsb_loss_and_grad import loss


jax.config.update('jax_enable_x64', True)


# arguments 1 and 2 have to be static because they are np.ndarray
val_and_grad = jax.jit(jax.value_and_grad(loss), static_argnums=tuple(np.arange(3, 13)))


def wrapper_loss_and_grad(*args):
    val, grad = val_and_grad(*args)
    return val, np.array(grad)


def unpack_dict(dict, *keys):
    vars = []
    for key in keys:
        vars.append(dict[key])
    return vars


def generate_and_preprocess(**params):
    # generate data
    m, p, n_concat, n, max_dilation, max_shift, noise_data, random_state, generation_function = unpack_dict(
        params, "m", "p", "n_concat", "n", "max_dilation", "max_shift", "noise_data", "random_state",
        "generation_function")
    rng = np.random.RandomState(random_state)

    if generation_function == 1:
        X_list, A_list, dilations, shifts, S_list, S = generate_data(
            m=m,
            p=p,
            n=n,
            max_shift=max_shift,
            max_dilation=max_dilation,
            noise_data=noise_data/5,
            n_bins=10,
            freq_level=50,
            S1_S2_scale=0.7,
            rng=rng,
            n_concat=n_concat,
        )
    else:
        X_list, A_list, dilations, shifts, S_list, _ = generate_new_data(
            m=m,
            p=p,
            n_concat=n_concat,
            n=n,
            max_dilation=max_dilation,
            max_shift=max_shift,
            noise_data=noise_data,
            rng=rng,
        )

    # center dilations and shifts
    dilations_c = dilations - np.mean(dilations, axis=0)
    shifts_c = shifts - np.mean(shifts, axis=0)

    # shift scale and dilation scale
    dilation_scale_per_source = params["dilation_scale_per_source"]

    if max_shift > 0:
        shift_scale = W_scale / max_shift  # scalar
    else:
        shift_scale = 1.
    if max_dilation > 1:
        dilation_scale = W_scale / (max_dilation - 1)  # scalar
        if dilation_scale_per_source:
            lambdas = np.array([[compute_lambda(s, n_concat=n_concat) for s in S] for S in S_list])
            dilation_scale *= lambdas  # (m, p) matrix
    else:
        dilation_scale = 1.

    # max_dilation_2 and max_shift_2
    bounds_factor = params["bounds_factor"]
    max_dilation_2 = 1 + bounds_factor * (max_dilation - 1)
    max_shift_2 = bounds_factor * max_shift

    # initialize W_list and S_list with permica
    max_delay = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay * n).astype("int")
    _, W_list_init, _, _ = permica(
        X_list, max_iter=1000, random_state=random_state, tol=1e-9,
        optim_delays=True, max_delay=max_delay_samples)
    S_list_init = np.array([np.dot(W, X) for W, X in zip(W_list_init, X_list)])
    S_list_init, W_list_init, dilations_init, shifts_init = permica_preprocessing(
        S_list_init, W_list_init, max_dilation, max_shift, n_concat, nb_points_grid=10, S_list_true=S_list)

    # center time parameters
    dilations_init_c = dilations_init - np.mean(dilations_init, axis=0) + 1
    shifts_init_c = shifts_init - np.mean(shifts_init, axis=0)

    # initialize A and B
    A_B_init_permica = params["A_B_init_permica"]
    if A_B_init_permica:
        A_init = dilations_init_c * dilation_scale
        B_init = shifts_init_c * shift_scale
    else:
        A_init = jnp.ones((m, p)) * dilation_scale
        B_init = jnp.zeros((m, p))
    W_A_B_init = jnp.concatenate([jnp.ravel(W_list_init), jnp.ravel(A_init), jnp.ravel(B_init)])

    # arguments used by LBFGSB
    noise_model, number_of_filters_squarenorm_f, filter_length_squarenorm_f = unpack_dict(
        params, "noise_model", "number_of_filters_squarenorm_f", "filter_length_squarenorm_f")
    use_envelop_term, number_of_filters_envelop, filter_length_envelop = unpack_dict(
        params, "use_envelop_term", "number_of_filters_envelop", "filter_length_envelop")

    args_lbfgsb = (
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
    )

    output = {
        "args_lbfgsb": args_lbfgsb,
        "max_dilation_2": max_dilation_2,
        "dilation_scale": dilation_scale,
        "max_shift_2": max_shift_2,
        "shift_scale": shift_scale,
        "dilations_c": dilations_c,
        "shifts_c": shifts_c,
        "A_list": A_list,
        "X_list": X_list,
        "W_list_init": W_list_init,
        "random_state": random_state,
        "max_delay_samples": max_delay_samples,
    }
    return output


def get_bounds(
    m, p, dilation_scale_per_source, max_dilation, dilation_scale, max_shift, shift_scale
):
    bounds_W = [(-jnp.inf, jnp.inf)] * (m * p**2)
    if dilation_scale_per_source:
        bounds_A = [
            (
                1 / max_dilation * dilation_scale.ravel()[i],
                max_dilation * dilation_scale.ravel()[i],
            )
            for i in range(m * p)
        ]
    else:
        bounds_A = [
            (1 / max_dilation * dilation_scale, max_dilation * dilation_scale)
        ] * (m * p)
    bounds_B = [(-max_shift * shift_scale, max_shift * shift_scale)] * (m * p)
    bounds_W_A_B = jnp.array(bounds_W + bounds_A + bounds_B)

    return bounds_W_A_B


def compute_3_scores(
    res_lbfgsb, m, p, dilation_scale, shift_scale, dilations_true_c, shifts_true_c, A_list_true
):
    # unpack results from LBFGSB
    W_A_B = res_lbfgsb[0]
    W_list_lbfgsb = W_A_B[:m*p**2].reshape((m, p, p))
    dilations_lbfgsb = 1 / (W_A_B[m*p**2: m*p*(p+1)].reshape((m, p)) / dilation_scale)
    shifts_lbfgsb = -W_A_B[m*p*(p+1):].reshape((m, p)) / shift_scale

    # center and reduce dilations and shifts
    dilations_lbfgsb_c = dilations_lbfgsb - np.mean(dilations_lbfgsb, axis=0)
    shifts_lbfgsb_c = shifts_lbfgsb - np.mean(shifts_lbfgsb, axis=0)

    # score on dilations and shifts
    score_dilations = np.mean((dilations_lbfgsb_c - dilations_true_c) ** 2)
    score_shifts = np.mean((shifts_lbfgsb_c - shifts_true_c) ** 2)

    # Amari distance
    amari_lbfgsb = np.mean([amari_distance(W, A) for W, A in zip(W_list_lbfgsb, A_list_true)])

    return score_dilations, score_shifts, amari_lbfgsb


def run_experiment(**params):
    # generate data, intialize with permica, find orders and signs
    start = time()
    data = generate_and_preprocess(**params)
    time_preprocess = time() - start

    # jit
    args_lbfgsb = data["args_lbfgsb"]
    start = time()
    wrapper_loss_and_grad(*args_lbfgsb)
    time_jit = time() - start

    # bounds and callback for LBFGSB
    m, p, dilation_scale_per_source, max_shift = unpack_dict(
        params, "m", "p", "dilation_scale_per_source", "max_shift")
    max_dilation_2, dilation_scale, max_shift_2, shift_scale = unpack_dict(
        data, "max_dilation_2", "dilation_scale", "max_shift_2", "shift_scale")

    bounds_W_A_B = get_bounds(
        m=m, p=p, dilation_scale_per_source=dilation_scale_per_source, max_dilation=max_dilation_2,
        dilation_scale=dilation_scale, max_shift=max_shift_2, shift_scale=shift_scale)

    # LBFGSB
    print("LBFGSB...")
    start = time()
    res_lbfgsb = fmin_l_bfgs_b(
        func=wrapper_loss_and_grad,
        x0=args_lbfgsb[0],
        args=args_lbfgsb[1:],
        bounds=bounds_W_A_B,
        disp=False,
        factr=1e3,
        pgtol=1e-5,
        maxiter=3000,
    )
    time_lbfgsb = time() - start
    time_lbfgsb += time_jit + time_preprocess
    print("LBFGSB done.")

    # get scores
    dilations_c, shifts_c, A_list = unpack_dict(data, "dilations_c", "shifts_c", "A_list")
    score_dilations_lbfgsb, score_shifts_lbfgsb, amari_lbfgsb = compute_3_scores(
        res_lbfgsb=res_lbfgsb, m=m, p=p, dilation_scale=dilation_scale, shift_scale=shift_scale,
        dilations_true_c=dilations_c, shifts_true_c=shifts_c, A_list_true=A_list)

    # MVICAD max_delay=max_shift
    n, random_state = unpack_dict(params, "n", "random_state")
    X_list, W_list_init = unpack_dict(data, "X_list", "W_list_init")
    start = time()
    _, W_mvicad, _, _, _, _ = multiviewica_delay(
        X_list,
        init=W_list_init,
        shared_delays=False,
        max_delay=int(max_shift*n),
        random_state=random_state,
        continuous_delays=False,
    )
    time_mvicad = time() - start
    amari_mvicad = np.mean([amari_distance(W, A) for W, A in zip(W_mvicad, A_list)])

    # MVICAD max_delay=max_delay
    max_delay_samples = data["max_delay_samples"]
    start = time()
    _, W_mvicad_2, _, _, _, _ = multiviewica_delay(
        X_list,
        init=W_list_init,
        shared_delays=False,
        max_delay=max_delay_samples,
        random_state=random_state,
        continuous_delays=False,
    )
    time_mvicad_2 = time() - start
    amari_mvicad_2 = np.mean([amari_distance(W, A) for W, A in zip(W_mvicad_2, A_list)])

    # Amari distance after permica
    amari_permica = np.mean([amari_distance(W, A) for W, A in zip(W_list_init, A_list)])

    # output
    output = {"Amari LBFGSB": amari_lbfgsb,
              "Dilations score LBFGSB": score_dilations_lbfgsb,
              "Shifts score LBFGSB": score_shifts_lbfgsb,
              "Time LBFGSB": time_lbfgsb,
              "Amari MVICAD": amari_mvicad,
              "Time MVICAD": time_mvicad,
              "Amari MVICAD ext": amari_mvicad_2,
              "Time MVICAD ext": time_mvicad_2,
              "Amari permica": amari_permica,
              "W scale": W_scale,
              "random state": random_state}
    return output


if __name__ == '__main__':
    # fixed params
    params = {
        "m": 5,
        "p": 3,
        "n_concat": 5,
        "n": 600,
        "max_shift": 0.05,
        "max_dilation": 1.15,
        "bounds_factor": 1.2,
        "noise_data": 0.05,
        "noise_model": 1,  # 1 by default
        "number_of_filters_squarenorm_f": 0,
        "filter_length_squarenorm_f": 3,
        "use_envelop_term": True,
        "number_of_filters_envelop": 1,
        "filter_length_envelop": 10,
        "dilation_scale_per_source": True,
        "generation_function": 2,
        "A_B_init_permica": True,
    }

    # varying params
    W_scales = np.logspace(0, 2, 20)
    nb_seeds = 30
    random_states = np.arange(nb_seeds)
    nb_expes = len(W_scales) * len(random_states)

    # run experiment
    print("\n############################################### Start ###############################################")
    df_res = pd.DataFrame()
    for i, (W_scale, random_state) in tqdm(enumerate(product(W_scales, random_states))):
        params["W_scale"] = W_scale
        params["random_state"] = random_state
        dict_expe = run_experiment(**params)
        df = pd.DataFrame(dict_expe, index=[i])
        df_res = pd.concat([df_res, df], ignore_index=True)
        print(f"Total number of experiments : {nb_expes}\n")
    print("\n######################################### Obtained DataFrame #########################################")
    print(df_res)

    # save dataframe
    results_dir = "/storage/store2/work/aheurteb/mvicad/lbfgsb_results/results_run_many_seeds/"
    save_name = f"DataFrame_with_{nb_seeds}_seeds_gen2_initpermica"
    save_path = results_dir + save_name
    df_res.to_csv(save_path, index=False)
    print("\n################################################ End ################################################")

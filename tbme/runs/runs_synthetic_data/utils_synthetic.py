import numpy as np
from picard import amari_distance
from time import time
from multiviewica import multiviewica, groupica
from multiviewica_delay import mvica_s, mvica_ds
from multiviewica_delay.multiviewica_dilations_shifts import generate_data


def compute_3_scores(
    dilations,
    shifts,
    W_list,
    dilations_true,
    shifts_true,
    A_list_true,
    max_dilation=1.15,
    max_shift=0.05,
):
    # center and reduce dilations and shifts
    dilations_c = dilations - np.mean(dilations, axis=0)
    shifts_c = shifts - np.mean(shifts, axis=0)
    dilations_true_c = dilations_true - np.mean(dilations_true, axis=0)
    shifts_true_c = shifts_true - np.mean(shifts_true, axis=0)
    # project onto [0, 1]
    dilations_p = (dilations_c - 1 / max_dilation) / (max_dilation - 1 / max_dilation)
    shifts_p = (shifts_c + max_shift) / (2 * max_shift)
    dilations_true_p = (dilations_true_c - 1 / max_dilation) / (max_dilation - 1 / max_dilation)
    shifts_true_p = (shifts_true_c + max_shift) / (2 * max_shift)
    # score on dilations and shifts
    error_dilations = np.mean(np.abs(dilations_p - dilations_true_p))
    error_shifts = np.mean(np.abs(shifts_p - shifts_true_p))
    # Amari distance
    amari = np.mean([amari_distance(W, A) for W, A in zip(W_list, A_list_true)])
    return error_dilations, error_shifts, amari


def run_experiment(
    m,
    p,
    n_concat,
    n,
    max_dilation=1.15,
    max_shift=0.05,
    noise_data=0.05,
    noise_model=1,
    n_bins=10,
    freq_level=50,
    S1_S2_scale=0.7,
    number_of_filters_squarenorm_f=0,
    filter_length_squarenorm_f=3,
    use_envelop_term=True,
    number_of_filters_envelop=1,
    filter_length_envelop=10,
    dilation_scale_per_source=True,
    W_scale=15,
    penalization_scale=1,
    random_state=None,
    nb_points_grid_init=10,
    verbose=False,
    return_all_iterations=True,
    dict_varying_outputs=None,
    max_iter=3000,
    onset=0,
):
    rng = np.random.RandomState(random_state)

    # generate data
    X_list, A_list, dilations, shifts, S_list, S = generate_data(
        m=m,
        p=p,
        n_concat=n_concat,
        n=n,
        max_dilation=max_dilation,
        max_shift=max_shift,
        noise_data=noise_data,
        n_bins=n_bins,
        freq_level=freq_level,
        S1_S2_scale=S1_S2_scale,
        rng=rng,
        onset=onset,
    )

    # LBFGSB
    print("LBFGSB...")
    start = time()
    W_list_lbfgsb, dilations_lbfgsb, shifts_lbfgsb, _, _, W_list_permica, _, _ = mvica_ds(
        X_list=X_list,
        n_concat=n_concat,
        max_dilation=max_dilation,
        max_shift=max_shift,
        dilation_scale_per_source=dilation_scale_per_source,
        W_scale=W_scale,
        penalization_scale=penalization_scale,
        random_state=random_state,
        noise_model=noise_model,
        number_of_filters_envelop=number_of_filters_envelop,
        filter_length_envelop=filter_length_envelop,
        number_of_filters_squarenorm_f=number_of_filters_squarenorm_f,
        filter_length_squarenorm_f=filter_length_squarenorm_f,
        use_envelop_term=use_envelop_term,
        max_iter=max_iter,
        nb_points_grid_init=nb_points_grid_init,
        verbose=verbose,
        return_all_iterations=return_all_iterations,
        S_list_true=S_list,
        onset=onset,
    )
    time_lbfgsb = time() - start
    print("LBFGSB done.")

    # get scores
    error_dilations_lbfgsb, error_shifts_lbfgsb, amari_lbfgsb = compute_3_scores(
        dilations=1/dilations_lbfgsb, shifts=-shifts_lbfgsb, W_list=W_list_lbfgsb,
        dilations_true=dilations, shifts_true=shifts, A_list_true=A_list)

    # MVICAD max_delay=max_shift
    start = time()
    _, W_mvicad, _, _, _, _ = mvica_s(
        X_list,
        init=W_list_permica,
        shared_delays=False,
        max_delay=int(max_shift*n),
        random_state=random_state,
        max_iter=max_iter,
        continuous_delays=False,
    )
    time_mvicad = time() - start
    amari_mvicad = np.mean([amari_distance(W, A) for W, A in zip(W_mvicad, A_list)])

    # MVICAD max_delay=max_delay
    max_delay_time = (1 + max_shift) * max_dilation - 1
    max_delay_samples = np.ceil(max_delay_time * n).astype("int")
    start = time()
    _, W_mvicad_2, _, _, _, _ = mvica_s(
        X_list,
        init=W_list_permica,
        shared_delays=False,
        max_delay=max_delay_samples,
        random_state=random_state,
        max_iter=max_iter,
        continuous_delays=False,
    )
    time_mvicad_2 = time() - start
    amari_mvicad_2 = np.mean([amari_distance(W, A) for W, A in zip(W_mvicad_2, A_list)])

    # MVICA
    start = time()
    _, W_mvica, _ = multiviewica(
        X_list,
        init=W_list_permica,
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-3,
    )
    time_mvica = time() - start
    amari_mvica = np.mean([amari_distance(W, A) for W, A in zip(W_mvica, A_list)])

    # GroupICA
    start = time()
    _, W_groupica, _ = groupica(
        X_list,
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-7,
    )
    time_groupica = time() - start
    amari_groupica = np.mean([amari_distance(W, A) for W, A in zip(W_groupica, A_list)])

    # Amari distance after permica
    amari_permica = np.mean([amari_distance(W, A) for W, A in zip(W_list_permica, A_list)])

    # output
    output = {"Amari LBFGSB": amari_lbfgsb,
              "Dilations error LBFGSB": error_dilations_lbfgsb,
              "Shifts error LBFGSB": error_shifts_lbfgsb,
              "Time LBFGSB": time_lbfgsb,
              "Amari MVICAD": amari_mvicad,
              "Time MVICAD": time_mvicad,
              "Amari MVICAD ext": amari_mvicad_2,
              "Time MVICAD ext": time_mvicad_2,
              "Amari MVICA": amari_mvica,
              "Time MVICA": time_mvica,
              "Amari GroupICA": amari_groupica,
              "Time GroupICA": time_groupica,
              "Amari permica": amari_permica,
              "random_state": random_state}
    if dict_varying_outputs is not None:
        output.update(dict_varying_outputs)
    return output

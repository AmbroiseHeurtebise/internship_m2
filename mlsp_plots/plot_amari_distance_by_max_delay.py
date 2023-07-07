import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed, Memory
from picard import amari_distance
from independent_vector_analysis import iva_g
from multiviewica_delay import (
    multiviewica,
    multiviewica_delay,
    generate_data,
    data_generation,
    data_generation_pierre,
    permica,
)


mem = Memory(".")


@mem.cache
def run_experiment(
    m,
    p,
    n,
    nb_intervals,
    nb_freqs,
    treshold,
    algo,
    max_delay,
    noise,
    random_state,
    shared_delays,
    generation_function,
    n_concat,
    n_bins,
    n_samples_per_interval,
    freq_level,
):
    if generation_function == 'first':
        X_list, A_list, _, _, _ = generate_data(
            m=m,
            p=p,
            n=n,
            nb_intervals=nb_intervals,
            nb_freqs=nb_freqs,
            treshold=treshold,
            delay=max_delay,
            noise=noise,
            random_state=random_state,
            shared_delays=shared_delays
        )
    elif generation_function == 'second':
        X_list, A_list, _, _, _ = data_generation(
            m=m,
            p=p,
            n=n,
            max_delay=max_delay,
            noise=noise,
            shared_delays=shared_delays,
            random_state=random_state,
            n_concat=n_concat,
        )
    elif generation_function == 'third':
        X_list, A_list, _, _, _ = data_generation_pierre(
            n_subjects=m,
            n_sources=p,
            n_bins=n_bins,
            n_samples_per_interval=n_samples_per_interval,
            freq_level=freq_level,
            max_delay=max_delay,
            noise=noise,
            shared_delays=shared_delays,
            random_state=random_state,
        )
    else:
        raise ValueError("Wrong generation function name")

    if algo == 'MVICA':
        _, W_list, _ = multiviewica(
            X_list,
            random_state=random_state
        )
    elif algo == 'MVICAD_one_delay':
        _, W_list, _, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=max_delay,
            every_n_iter_delay=2,
            random_state=random_state,
            shared_delays=True,
        )
    elif algo == 'MVICAD_fixed_maxdelay':
        _, W_list, _, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=10,
            every_n_iter_delay=2,
            random_state=random_state,
            shared_delays=False,
        )
    elif algo == 'MVICAD':
        _, W_list, _, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=max_delay,
            every_n_iter_delay=2,
            random_state=random_state,
            shared_delays=False,
        )
    elif algo == 'MVICAD_test_alex':
        _, W_list, _, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=max_delay,
            every_n_iter_delay=2,
            random_state=random_state,
            shared_delays=False,
            test_alex=True,
        )
    elif algo == 'MVICAD_with_f':
        _, W_list, _, _, _, _ = multiviewica_delay(
            X_list,
            max_delay=max_delay,
            every_n_iter_delay=2,
            random_state=random_state,
            shared_delays=False,
            optim_delays_with_f=True,
        )
    elif algo == 'permica':
        _, W_list, _, _ = permica(
            X_list,
            random_state=random_state,
        )
    elif algo == 'IVA':
        X_list_reshaped = X_list.transpose(1, 2, 0)
        W_list, cost, Sigma_n, isi = iva_g(X_list_reshaped, jdiag_initW=False)
        W_list = W_list.transpose(2, 0, 1)
    else:
        raise ValueError("Wrong algo name")
    amari = np.sum([amari_distance(W, A) for W, A in zip(W_list, A_list)])
    output = {"Algo": algo, "Delay": max_delay,
              "random_state": random_state, "Amari_distance": amari}
    return output


if __name__ == '__main__':
    # Parameters
    m = 5
    p = 2
    n = 600
    nb_intervals = 5  # only serves if generation_function == "first"
    nb_freqs = 10  # only serves if generation_function == "first"
    treshold = 1  # only serves if generation_function == "first"
    n_concat = 3  # only serves if generation_function == "second"
    n_bins = 10  # only serves if generation_function == "third"
    n_samples_per_interval = n // n_bins  # only serves if generation_function == "third"
    freq_level = 40  # only serves if generation_function == "third"
    algos = ['MVICAD', 'MVICAD_test_alex']
    delays = np.linspace(0, 10, 11, dtype=int)
    n_expe = 100
    shared_delays = False
    generation_function = 'third'
    if generation_function == 'first':
        noise = 1
    elif generation_function == 'second':
        # noise = 2 * 1e-4  # if we don't use sources_generation_zero_mean()
        noise = 0.03
    elif generation_function == 'third':
        noise = 0.5
    else:
        raise ValueError("Wrong generation function name")
    N_JOBS = 8

    # Run ICA
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(
            m, p, n, nb_intervals, nb_freqs, treshold, algo, max_delay, noise,
            random_state, shared_delays, generation_function, n_concat, n_bins,
            n_samples_per_interval, freq_level)
        for algo, max_delay, random_state
        in product(algos, delays, range(n_expe))
    )
    results = pd.DataFrame(results).drop(columns='random_state')

    # Plot
    fig = plt.figure(figsize=(8, 4))
    sns.set(font_scale=1.5)  # XXX
    sns.set_style("white")
    sns.set_style('ticks')
    sns.lineplot(
        data=results, x="Delay", y="Amari_distance", hue="Algo", linewidth=2.5)
    plt.yscale("log")
    fontsize = 28
    x_ = plt.xlabel("Delay (ms)", fontsize=fontsize)
    y_ = plt.ylabel("Amari distance", fontsize=fontsize)
    leg = plt.legend(
        prop={'size': 18}, loc='upper right', bbox_to_anchor=(0.57, 1.036))
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Synthetic experiment", fontsize=fontsize)
    if shared_delays:
        shared_name = ""
    else:
        shared_name = "multiple_"
    if generation_function == 'first':
        gen_name = ""
    elif generation_function == 'second':
        gen_name = "_new_gen"
    elif generation_function == 'third':
        gen_name = "_gen_pierre"
    save_name = "amari_without_borders_seeds%s_m%s_p%s_nbins%s_freqs%s_noise%s.pdf" % (n_expe, m, p, n_bins, freq_level, noise)
    # save_name = "amari_by_delay_%s%s_seeds%s_m%s_p%s_nbins%s_freqs%s_noise%s.pdf" % (shared_name, n_expe, gen_name, m, p, n_bins, freq_level, noise)
    # save_name = "amari_fixed_maxdelay_seeds%s_m%s_p%s_nbins%s_freqs%s_noise%s.pdf" % (n_expe, m, p, n_bins, freq_level, noise)
    # max_delay = np.max(delays)
    # save_name = "amari_with_IVA_seeds%s_m%s_p%s_nbins%s_freqs%s_noise%s_maxdelay%s.pdf" % (n_expe, m, p, n_bins, freq_level, noise, max_delay)
    plt.savefig(
        "mlsp_figures/" + save_name, bbox_extra_artists=[x_, y_],
        bbox_inches="tight")

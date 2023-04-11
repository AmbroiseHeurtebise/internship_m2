import numpy as np
from time import time
from tqdm import tqdm
import os
from multiviewica_delay import (
    multiviewica_delay,
    _apply_delay,
    _apply_delay_by_source,
)


def do_extract_data(
    task="visual",
    artificial_delays=False,
    shared_delays=False,
    n_comp=20,
    max_delay=40,
    seed=0,
    subsample=None,
):
    # prepare results folder
    if artificial_delays:
        artificial_name = "_artificial_"
    else:
        artificial_name = "_"
    if shared_delays:
        shared_name = ""
    else:
        shared_name = "multiple_"
    if subsample:
        subsample_name = "_subsample" + str(subsample)
    else:
        subsample_name = ""
    algo = "mvicad_" + task + artificial_name + shared_name + "p" + str(n_comp) + subsample_name
    save_path = "/storage/store2/work/aheurteb/mvicad/mlsp_results/%s/" % algo
    os.makedirs(save_path, exist_ok=True)

    # load dataset
    rng = np.random.RandomState(seed)
    dataset_path = "/storage/store2/work/aheurteb/mvicad/data/"
    if task == "visual":
        if artificial_delays:
            data = np.load(
                dataset_path + "X_visual_task_mag_477_artificially_delayed.npy")
        else:
            data = np.load(dataset_path + "X_visual_task_mag_477.npy")
        if subsample is not None:
            subjects = rng.choice(len(data), subsample, replace=False)
            data = data[subjects]

    elif task == "auditory":
        if artificial_delays:
            data = np.load(
                dataset_path + "X_auditory_task_mag_501_artificially_delayed.npy")
        else:
            data = np.load(dataset_path + "X_auditory_task_mag_501.npy")
        if subsample is not None:
            subjects = rng.choice(len(data), subsample, replace=False)
            data = data[subjects]

    else:
        raise ValueError("Wrong task name")

    # preprocessing : PCA
    X = []
    for d in tqdm(data):
        _, D, v = np.linalg.svd(d, full_matrices=False)
        y = v[:n_comp]
        x = y * D[:n_comp, None]
        X.append(x)
    X = np.array(X)
    print(X.shape)

    # MVICAD
    t0 = time()
    _, W_list, Y_avg, tau_list, _ = multiviewica_delay(
        X,
        tol=1e-4,
        max_iter=1000,
        max_delay=max_delay,
        every_n_iter_delay=10,
        shared_delays=shared_delays,
        random_state=rng,
        verbose=True,
    )
    dt = time() - t0

    # save data
    np.save(save_path + "W.npy", W_list)
    np.save(save_path + "shared_sources.npy", Y_avg)
    np.save(save_path + "tau_list.npy", tau_list)
    np.save(save_path + "fit_time.npy", dt)
    if subsample is not None:
        np.save(save_path + "subjects.npy", subjects)

    # compute derivatives
    Y_list = [np.dot(w, x) for w, x in zip(W_list, X)]
    if shared_delays:
        Y_list = _apply_delay(Y_list, -tau_list)
    else:
        Y_list = _apply_delay_by_source(Y_list, -tau_list)  # XXX
    Y_avg /= np.linalg.norm(Y_avg, axis=1, keepdims=1)
    signs = 2 * ((np.max(Y_avg, axis=1) + np.min(Y_avg, axis=1) > 0) - 0.5)
    Y_avg *= signs[:, None]
    order = np.argsort(np.argmax(Y_avg, axis=1))
    W_list = np.array([(w * signs[:, None])[order] for w in W_list])
    Y_list = np.array([(y * signs[:, None])[order] for y in Y_list])

    np.save(save_path + "W_derivatives.npy", W_list)  # order has changed
    np.save(save_path + "shared_sources_derivatives.npy", Y_avg)  # norm and sign have changed
    np.save(save_path + "sources_derivatives.npy", Y_list)
    np.save(
        save_path + "source_reduced_derivatives.npy", np.mean(Y_list, axis=0))


if __name__ == '__main__':
    # parameters
    task = "visual"
    artificial_delays = False
    shared_delays = False
    n_comp = 10
    max_delay = 40
    seed = 1
    subsample = None

    # apply mvicad and save results
    do_extract_data(
        task,
        artificial_delays,
        shared_delays,
        n_comp,
        max_delay,
        seed,
        subsample,
    )

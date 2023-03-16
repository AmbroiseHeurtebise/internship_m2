import numpy as np
from time import time
from tqdm import tqdm
import os
from multiviewica_delay import (
    multiviewica,
    permica,
    groupica,
    multiviewica_delay
)


def do_extract_data(
    path,
    algo,
    save_path,
    # keep_path,
    seed,
    mvica_path="/storage/store2/work/hrichard/mvica/",
    subsample=None,
):
    os.makedirs(save_path, exist_ok=True)
    rng = np.random.RandomState(seed)
    n_comp = 20
    n_concat = 2
    save_str = "%d_%d_%d_%s_%s" % (496, 10, 2, "audio", "task",)
    subjects = np.load(mvica_path + "subjects_%s.npy" % save_str)
    data = np.load(path)
    n_subjects = len(data)

    if subsample is not None:
        # Select a subset of 100 subjects
        subjects_indexes = rng.choice(np.arange(496), subsample, replace=False)
        save_str = "%d_%d_%d_%s_%s_%i_%s" % (
            subsample,
            10,
            2,
            "audio",
            "task",
            seed,
            algo,
        )
        subjects = subjects[subjects_indexes]
        data = np.load(path)[subjects_indexes]
        n_subjects = len(data)

    print(data.shape)
    ##### Preprocessing : PCA or SRM
    X = []
    W_srm = []
    for i, d in tqdm(enumerate(data)):
        u, D, v = np.linalg.svd(d, full_matrices=False)
        y = v[:n_comp]
        x = y * D[:n_comp, None]
        X.append(x)
        W_srm.append(u[:, :n_comp])
    X = np.array(X)
    _, _, n = X.shape
    W_srm = np.array(W_srm)
    S_srm = np.mean(X, axis=0)
    print(W_srm.shape)
    print(S_srm.shape)
    ########## Group ICA as init
    _, W_p, S_p = groupica(X, random_state=rng)
    # A_tp = [np.dot(W_s, np.linalg.pinv(W)) for W_s, W in zip(W_srm, W_p)]
    
    t0 = time()
    if algo == "permica":
        _, W_m, S_m = permica(X, random_state=rng)
    elif algo == "mvica":
        if os.path.isfile(save_path + "W_%s.npy" % save_str) and os.path.isfile(save_path + "shared_sources_%s.npy" % save_str):
            with open(save_path + "W_%s.npy" % save_str, "rb") as save_W_file:
                W_m = np.load(save_W_file)
            if os.path.isfile(save_path + "shared_sources_%s.npy" % save_str):
                with open(save_path + "shared_sources_%s.npy" % save_str, "rb") as save_S_file:
                    S_m = np.load(save_S_file)
            else:
                S_m = np.mean([np.dot(W, x) for W, x in zip(W_m, X)], axis=0)
                np.save(save_path + "shared_sources_%s.npy" % save_str, S_m)
        else:
            ########## Multi view ICA
            _, W_m, S_m = multiviewica(
                X,
                init=W_p,
                tol=1e-4,
                max_iter=10000,
                random_state=rng,
                verbose=True,
            )
            np.save(save_path + "W_%s.npy" % save_str, W_m)
            np.save(save_path + "shared_sources_%s.npy" % save_str, S_m)
    elif algo == "groupica":
        _, _, S_m = groupica(X, tol=1e-4, max_iter=10000, random_state=seed)
        W_m = [x.dot(np.linalg.pinv(S_m)) for x in X]
        np.save(save_path + "W_%s.npy" % save_str, W_m)
    elif algo == "mvicad_every_10_iter":
        delay_max = 20
        _, W_m, S_m, tau_list, _ = multiviewica_delay(
            X,
            init=W_p,
            tol=1e-4,
            max_iter=10000,
            optim_delays_ica=True,  # XXX
            delay_max=delay_max,
            every_N_iter_delay=10,  # XXX
            random_state=rng,
            verbose=True,
        )
        np.save(save_path + "W_%s.npy" % save_str, W_m)
        np.save(save_path + "shared_sources_%s.npy" % save_str, S_m)
        np.save(save_path + "tau_list_%s.npy" % save_str, tau_list)
    
    dt = time() - t0
    np.save(save_path + "fit_time_%s.npy" % save_str, dt)
    
    ######### Compute derivatives
    A_m = [np.linalg.pinv(W) for W in W_m]  # Total mixing
    A_tm = [np.dot(W_s, np.linalg.pinv(W)) for W_s, W in zip(W_srm, W_m)]
    Y_m = [np.dot(w, x) for w, x in zip(W_m, X)]  # Per subject sources
    Y_p = [np.dot(w, x) for w, x in zip(W_p, X)]
    S_m /= np.linalg.norm(S_m, axis=1, keepdims=1)  # Shared sources
    S_p /= np.linalg.norm(S_p, axis=1, keepdims=1)
    ### reorder and align
    signs = 2 * ((np.max(S_m, axis=1) + np.min(S_m, axis=1) > 0) - 0.5)
    S_m *= signs[:, None]
    S_srm *= ((np.max(S_srm, axis=1) + np.min(S_srm, axis=1)) > 0 - 0.5)[
        :, None
    ]
    S_p *= ((np.max(S_p, axis=1) + np.min(S_p, axis=1)) > 0 - 0.5)[:, None]
    n_ = n // n_concat
    order = np.argsort(np.argmax(S_m[:, :n_], axis=1))
    W_m = np.array([(w * signs[:, None])[order] for w in W_m])
    np.save(save_path + "W_%s_derivatives.npy" % save_str, W_m)
    Y_m = np.array([(y * signs[:, None])[order] for y in Y_m])
    A_tm = np.array(A_tm)[:, :, order] * signs[None, None, :]
    A_tm /= np.linalg.norm(A_tm, axis=1, keepdims=True)
    A_mean = np.mean(A_tm, axis=0)
    # np.save(
    #     keep_path + "camcan_%s_shared_sources_%s.npy" % (algo, save_str), S_m
    # )
    np.save(save_path + "shared_sources_%s_derivatives.npy" % save_str, S_m)
    np.save(save_path + "mixing_%s.npy" % save_str, A_tm)
    np.save(save_path + "sources_%s.npy" % save_str, Y_m)
    np.save(save_path + "source_reduced_%s.npy" % save_str, np.mean(Y_m, axis=0))
    np.save(save_path + "subjects_%s.npy" % save_str, subjects)


path = "/storage/store/work/hrichard/mvica/results_nostd/data_496_10_2_audio_task.npy"
algos = ["mvicad_every_10_iter"]
seeds = np.arange(1)
seed = seeds[0]
for algo in algos:
    do_extract_data(
        path,
        algo,
        "/storage/store2/work/aheurteb/mvicad/results/%s/" % algo,
        # "../results/",
        seed,
        subsample=100,
    )

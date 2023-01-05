import numpy as np
from amvica.mifast import mifast
from amvica.mifast2 import mifast2
from shica import shica_ml
from time import time
from amvica.gavica_em import gavica_em
from tqdm import tqdm
import os
from joblib import Parallel, delayed, dump
from amvica.core import amvica
from multiviewica import multiviewica, permica, groupica
from sklearn.utils.extmath import randomized_svd
from mvlearn.decomposition import GroupICA
from tqdm import tqdm


def do_extract_data(
    path,
    algo,
    save_path,
    keep_path,
    seed,
    mvica_path="/storage/store2/work/hrichard/mvica/",
    subsample=None,
):
    os.makedirs(save_path, exist_ok=True)
    rng = np.random.RandomState(seed)
    n_comp = 10
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
    n_comp = 10
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
    ########## Group ICA as init
    _, W_p, S_p = groupica(X, random_state=rng)
    A_tp = [np.dot(W_s, np.linalg.pinv(W)) for W_s, W in zip(W_srm, W_p)]
    t0 = time()
    if algo == "permica":
        _, W_m, S_m = permica(X, random_state=rng)
    elif algo == "groupica":
        W_m = W_p
        S_m = S_p
    elif algo == "mvica":
        ########## Multi view ICA
        _, W_m, S_m = multiviewica(
            X,
            init=W_p,
            tol=1e-4,
            max_iter=10000,
            random_state=rng,
            verbose=True,
        )
    # elif algo == "amvica":
    #     W_m, Sigmas, S_m = shica_ml(X, 10000, verbose=True)
    #     np.save(save_path + "W_%s.npy" % save_str, W_m)
    #     np.save(save_path + "Sigmas_%s.npy" % save_str, Sigmas)
    elif algo == "gavica":
        W_m, L, sigmas, S_m = gavica_em(X, W_p, 10000, verbose=True, tol=1e-4)
        np.save(save_path + "W_%s.npy" % save_str, W_m)
        np.save(save_path + "L_%s.npy" % save_str, L)
        np.save(save_path + "sigmas_%s.npy" % save_str, sigmas)
    elif algo == "mifast":
        W_m, Sigmas, S_m = mifast(X, 10000, verbose=True, tol=1e-4)
        np.save(save_path + "W_%s.npy" % save_str, W_m)
        np.save(save_path + "Sigmas_%s.npy" % save_str, Sigmas)
    elif algo == "mifast2":
        W_m, Sigmas, S_m = mifast2(X, verbose=True)
        np.save(save_path + "W_%s.npy" % save_str, W_m)
        np.save(save_path + "Sigmas_%s.npy" % save_str, Sigmas)
    elif algo == "groupica":
        _, _, S_m = groupica(X, tol=1e-4, max_iter=10000, random_state=seed)
        W_m = [x.dot(np.linalg.pinv(S_m)) for x in X]
        np.save(save_path + "W_%s.npy" % save_str, W_m)
    elif algo == "extendedgroupica":
        S_m = (
            GroupICA(multiview_output=False, random_state=seed,)
            .fit_transform(np.array([x.T for x in X]))
            .T
        )
        W_m = [x.dot(np.linalg.pinv(S_m)) for x in X]
        np.save(save_path + "W_%s.npy" % save_str, W_m)
    elif algo == "canica":
        S_m = (
            GroupICA(
                multiview_output=False,
                prewhiten=True,
                random_state=seed,
                ica_kwargs={"ortho": False, "extended": False},
            )
            .fit_transform(np.array([x.T for x in X]))
            .T
        )
        W_m = [x.dot(np.linalg.pinv(S_m)) for x in X]
        np.save(save_path + "W_%s.npy" % save_str, W_m)
    elif algo == "extendedcanica":
        S_m = (
            GroupICA(
                multiview_output=False,
                prewhiten=True,
                random_state=seed,
                ica_kwargs={"verbose": True},
            )
            .fit_transform(np.array([x.T for x in X]))
            .T
        )
        W_m = [x.dot(np.linalg.pinv(S_m)) for x in X]
        np.save(save_path + "W_%s.npy" % save_str, W_m)
    elif algo == "multisetcca":
        W_m, Sigmas, S_m = mifast2(X, use_jointdiag=False)
        np.save(save_path + "W_%s.npy" % save_str, W_m)
        np.save(save_path + "Sigmas_%s.npy" % save_str, Sigmas)
    elif algo == "cansubgaussica":
        S_m = (
            GroupICA(
                multiview_output=False,
                prewhiten=True,
                ica_kwargs={"ortho": False, "extended": False, "fun": "cube"},
                random_state=seed,
            )
            .fit_transform(np.array([x.T for x in X]))
            .T
        )
        W_m = np.array([x.dot(np.linalg.pinv(S_m)) for x in X])
        np.save(save_path + "W_%s.npy" % save_str, W_m)

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
    # np.save(save_path + "W_%s.npy" % save_str, W_m)
    Y_m = np.array([(y * signs[:, None])[order] for y in Y_m])
    A_tm = np.array(A_tm)[:, :, order] * signs[None, None, :]
    A_tm /= np.linalg.norm(A_tm, axis=1, keepdims=True)
    A_mean = np.mean(A_tm, axis=0)
    np.save(
        keep_path + "camcan_%s_shared_sources_%s.npy" % (algo, save_str), S_m
    )
    np.save(save_path + "mixing_%s.npy" % save_str, A_tm)
    np.save(save_path + "sources_%s.npy" % save_str, Y_m)
    np.save(
        save_path + "source_reduced_%s.npy" % save_str, np.mean(Y_m, axis=0)
    )
    np.save(save_path + "subjects_%s.npy" % save_str, subjects)


path = "/storage/store/work/hrichard/mvica/results_nostd/data_496_10_2_audio_task.npy"

from joblib import Parallel, delayed

# algos = [
#     "groupica",
#     "extendedgroupica",
#     "canica",
#     "extendedcanica",
#     "cansubgaussica",
#     "gavica",
#     "mvica",
#     "amvica",
# ]
# algos = ["mifast2"]
# algos = ["mifast2"]
# seeds = np.arange(10)
# Parallel(n_jobs=20, verbose=True)(
#     delayed(do_extract_data)(
#         path,
#         algo,
#         "/storage/store2/work/hrichard/%s/" % algo,
#         "../results/",
#         seed,
#         subsample=400,
#     )
#     for algo in algos
#     for seed in seeds
# )

algos = ["extendedcanica"]
seeds = np.arange(1)
seed = seeds[0]
for algo in algos:
    do_extract_data(
        path,
        algo,
        "/storage/store2/work/hrichard/%s/" % algo,
        "../results/",
        seed,
    )

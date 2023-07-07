import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiviewica_delay import multiviewica_delay, multiviewica


def get_losses(
    task="visual",
    seed=0,
    n_comp=20,  # nb components PCA
    max_delay=20,
    nb_subjects=None,
):
    rng = np.random.RandomState(seed)

    # Paths
    if task == "auditory":
        path = "/storage/store2/work/aheurteb/mvicad/data/X_auditory_task_mag_501.npy"
    elif task == "visual":
        path = "/storage/store2/work/aheurteb/mvicad/data/X_visual_task_mag_477.npy"
    else:
        raise ValueError("Wrong task name")

    # Load data
    evoked_data = np.load(path)
    print("{} task dataset shape : {}".format(task, evoked_data.shape))

    # PCA
    X = []
    for i, d in tqdm(enumerate(evoked_data)):
        u, D, v = np.linalg.svd(d, full_matrices=False)
        y = v[:n_comp]
        x = y * D[:n_comp, None]
        X.append(x)
    X = np.array(X)
    m, p, n = X.shape
    print("{} task dataset shape : {}".format(task, X.shape))

    # Select a subset of subjects
    if nb_subjects is not None:
        subjects = rng.choice(len(X), size=nb_subjects, replace=False)
        X_list = X[subjects]
    else:
        X_list = X
    print("X_list shape : {}".format(X_list.shape))

    # MVICAD multiple sources
    _, _, _, _, loss = multiviewica_delay(
        X_list,
        max_delay=max_delay,
        every_n_iter_delay=10,
        random_state=np.random.RandomState(seed),
        multiple_sources=True,
        return_loss=True,
        verbose=True
    )
    loss_mvicad_mul, _ = loss
    loss_mvicad_mul = np.array(loss_mvicad_mul)
    max_iter = len(loss_mvicad_mul) - 1

    # MVICAD multiple sources with f
    _, _, _, _, loss = multiviewica_delay(
        X_list,
        max_delay=max_delay,
        every_n_iter_delay=10,
        tol=1e-5,
        tol_init=1e-3,
        max_iter=max_iter,
        random_state=np.random.RandomState(seed),
        multiple_sources=True,
        optim_delays_with_f=True,
        return_loss=True,
        verbose=True
    )
    loss_mvicad_mul_with_f, _ = loss
    loss_mvicad_mul_with_f = np.array(loss_mvicad_mul_with_f)

    # MVICAD one source
    _, _, _, _, loss = multiviewica_delay(
        X_list,
        max_delay=max_delay,
        every_n_iter_delay=10,
        tol=1e-5,
        tol_init=1e-3,
        max_iter=max_iter,
        random_state=np.random.RandomState(seed),
        multiple_sources=False,
        return_loss=True,
        verbose=True
    )
    loss_mvicad, _ = loss
    loss_mvicad = np.array(loss_mvicad)

    # MVICAD one source with f
    _, _, _, _, loss = multiviewica_delay(
        X_list,
        max_delay=max_delay,
        every_n_iter_delay=10,
        tol=1e-5,
        tol_init=1e-3,
        max_iter=max_iter,
        random_state=np.random.RandomState(seed),
        multiple_sources=False,
        optim_delays_with_f=True,
        return_loss=True,
        verbose=True
    )
    loss_mvicad_with_f, _ = loss
    loss_mvicad_with_f = np.array(loss_mvicad_with_f)

    # MVICA
    _, _, _, loss_mvica = multiviewica(
        X_list,
        tol=1e-5,
        tol_init=1e-3,
        max_iter=max_iter+1,
        random_state=np.random.RandomState(seed),
        return_loss=True,
        verbose=True
    )

    output = {
        "mvica": loss_mvica,
        "mvicad": loss_mvicad,
        "mvicad with f": loss_mvicad_with_f,
        "mvicad multiple sources": loss_mvicad_mul,
        "mvicad multiple sources with f": loss_mvicad_mul_with_f
        }
    return output


if __name__ == '__main__':
    # parameters
    task = "visual"  # should be either "auditory" or "visual"
    seed = 1
    max_delay = 20
    nb_subjects = 20

    # get losses
    losses = get_losses(
        task=task,
        seed=seed,
        max_delay=max_delay,
        nb_subjects=nb_subjects
    )
    losses = pd.DataFrame(losses)

    # save results in a csv file
    save_path = "/storage/store2/work/aheurteb/mvicad/"
    name = "losses_%s_task_%s_subjects_seed_%s.pkl" % (task, nb_subjects, seed)
    with open(save_path + "mlsp_results/" + name, "wb") as save_losses_file:
        pickle.dump(losses, save_losses_file)

    # plot
    plt.plot(losses["mvica"], label="mvica")
    plt.plot(losses["mvicad"], label="mvicad")
    plt.plot(losses["mvicad with f"], label="mvicad with f")
    plt.plot(losses["mvicad multiple sources"], label="mvicad mul sources")
    plt.plot(losses["mvicad multiple sources with f"],
             label="mvicad mul sources with f")
    plt.title("Loss of MVICA and MVICAD ; seed = {}".format(seed))
    x_ = plt.xlabel("Iterations")
    y_ = plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        save_path + "mlsp_figures/losses_%s_task_%s_subjects_seed_%s.jpeg" % (task, nb_subjects, seed),
        bbox_extra_artists=[x_, y_], bbox_inches="tight")

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from joblib import Memory
import matplotlib.pyplot as plt
from multiviewica_delay import multiviewica_delay, multiviewica


mem = Memory(".")


@mem.cache
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
        tol=1e-2,  # XXX
        tol_init=1e-3,
        max_delay=max_delay,
        every_n_iter_delay=10,
        random_state=np.random.RandomState(seed),
        shared_delays=False,
        return_loss=True,
        verbose=True,
    )
    loss_mvicad_mul, _ = loss
    loss_mvicad_mul = np.array(loss_mvicad_mul)
    max_iter = len(loss_mvicad_mul)

    # MVICA
    _, _, _, loss_mvica = multiviewica(
        X_list,
        tol=1e-5,
        tol_init=1e-3,
        max_iter=max_iter,
        random_state=np.random.RandomState(seed),
        return_loss=True,
        verbose=True,
    )

    output = {
        "mvica": loss_mvica,
        "mvicad multiple sources": loss_mvicad_mul
        }
    return output


if __name__ == '__main__':
    # parameters
    task = "visual"  # should be either "auditory" or "visual"
    nb_seeds = 10
    max_delay = 20
    nb_subjects = 20

    # get losses
    losses_mvica = []
    losses_mvicad = []
    for seed in range(nb_seeds):
        losses = get_losses(
            task=task,
            seed=seed,
            max_delay=max_delay,
            nb_subjects=nb_subjects
        )
        losses = pd.DataFrame(losses)

        losses_mvica.append(losses['mvica'])
        losses_mvicad.append(losses['mvicad multiple sources'])

        # save results in a csv file
        save_path = "/storage/store2/work/aheurteb/mvicad/"
        name = "losses_%s_task_%s_subjects_seed_%s.pkl" % (task, nb_subjects, seed)
        with open(save_path + "mlsp_results/losses/" + name, "wb") as save_losses_file:
            pickle.dump(losses, save_losses_file)

    # plot
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(nb_seeds):
        divisor = -losses_mvica[i][0]
        if i == 0:
            plt.plot(losses_mvica[i] / divisor, label="MVICA", c=colors[0])
            plt.plot(losses_mvicad[i] / divisor, label="MVICAD", c=colors[1])
        else:
            plt.plot(losses_mvica[i] / divisor, c=colors[0])
            plt.plot(losses_mvicad[i] / divisor, c=colors[1])

    plt.title(f"{task} task \nLosses of MVICA and MVICAD for {nb_seeds} seeds")
    x_ = plt.xlabel("Iterations")
    y_ = plt.ylabel("Loss")
    ax.grid(True)
    ax.set_yticklabels([])
    plt.legend()
    plt.savefig(
        save_path + "mlsp_figures/losses_%s_task_%s_subjects_%s_seeds.jpeg" % (task, nb_subjects, nb_seeds),
        bbox_extra_artists=[x_, y_], bbox_inches="tight")

import numpy as np
from multiviewica import multiviewica
from utils_camcan import load_and_reduce_data


# parameters
task = "auditory"
n_concat = 1
n_components_pca = 10
n_subjects_subgroup = 100
random_state = 42

# load and reduce data
X, subjects, ages, n_subjects_data = load_and_reduce_data(
    task, n_concat, n_components_pca, n_subjects_subgroup, random_state)

# whitening of X
X = np.array([x - np.mean(x, axis=1, keepdims=True) for x in X])
X = np.array([x / np.linalg.norm(x, axis=1, keepdims=True) for x in X])

# MVICA
_, W_mvica, S_mvica = multiviewica(
    X,
    max_iter=1000,
    random_state=random_state,
    tol=1e-3,
)

# correct scale and sign
scale = np.linalg.norm(S_mvica, axis=1)  # shape (p,)
sign = 2 * ((np.max(S_mvica, axis=1) + np.min(S_mvica, axis=1) > 0) - 0.5)   # shape (p,)
scale_and_sign = (scale * sign)[:, None]  # shape (p, 1)
S_mvica /= scale_and_sign
W_mvica = np.array([W / scale_and_sign for W in W_mvica])

# save data
if n_subjects_subgroup is None:
    n_subjects_subgroup = n_subjects_data
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_camcan/mvica/clean_subjects/"
suffix = f"_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
np.save(results_dir + "W" + suffix, W_mvica)
np.save(results_dir + "S" + suffix, S_mvica)

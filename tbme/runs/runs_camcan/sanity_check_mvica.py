import numpy as np
from multiviewica import multiviewica
from utils_camcan import load_and_reduce_data


# parameters
task = "visual"
n_concat = 6
n_components_pca = 5
n_subjects_subgroup = 50
random_state = 42

# load and reduce data
X, subjects, ages, n_subjects_data = load_and_reduce_data(
    task, n_concat, n_components_pca, n_subjects_subgroup, random_state)

# MVICA
_, W_mvica, S_mvica = multiviewica(
    X,
    max_iter=1000,
    random_state=random_state,
    tol=1e-3,
)

# save data
if n_subjects_subgroup is None:
    n_subjects_subgroup = n_subjects_data
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_camcan/mvica/"
W_save_name = f"W_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
S_save_name = f"Y_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
np.save(results_dir + W_save_name, W_mvica)
np.save(results_dir + S_save_name, S_mvica)

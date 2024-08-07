import numpy as np
from multiviewica_delay import mvica_ds
from utils_camcan import load_and_reduce_data


# parameters
task = "auditory"
n_concat = 3
n_components_pca = 5
n_subjects_subgroup = 50  # None if whole dataset
max_dilation = 1.15
max_shift = 0.05
W_scale = 200
random_state = 42

# load and reduce data
X, _, ages = load_and_reduce_data(
    task, n_concat, n_components_pca, n_subjects_subgroup, random_state)

# whitening of X
X = np.array([x - np.mean(x, axis=1, keepdims=True) for x in X])
X = np.array([x / np.linalg.norm(x, axis=1, keepdims=True) for x in X])

# MVICAD^2
W_list, dilations, shifts, Y_list = mvica_ds(
    X_list=X,
    n_concat=n_concat,
    max_dilation=max_dilation,
    max_shift=max_shift,
    dilation_scale_per_source=True,
    W_scale=W_scale,
    penalization_scale=1,
    random_state=random_state,
    noise_model=1,
    number_of_filters_envelop=1,
    filter_length_envelop=5,
    number_of_filters_squarenorm_f=1,
    filter_length_squarenorm_f=2,
    use_envelop_term=True,
    nb_points_grid_init=10,
    verbose=True,
    return_all_iterations=False,
    S_list_true=None,
    factr=1e-1,  # instead of 1e5
    pgtol=1e-8,
)

# correct scale and sign
scale = np.linalg.norm(Y_list, axis=2)  # shape (m, p)
sign = 2 * ((np.max(Y_list, axis=2) + np.min(Y_list, axis=2) > 0) - 0.5)   # shape (m, p)
scale_and_sign = (scale * sign)[:, :, None]  # shape (m, p, 1)
W_list /= scale_and_sign
Y_list /= scale_and_sign

# save data
if n_subjects_subgroup is None:
    n_subjects_subgroup = len(X)
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_camcan/mvicad2/clean_subjects/"
suffix = f"_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
np.save(results_dir + "W" + suffix, W_list)
np.save(results_dir + "dilations" + suffix, dilations)
np.save(results_dir + "shifts" + suffix, shifts)
np.save(results_dir + "Y" + suffix, Y_list)
np.save(results_dir + "ages" + suffix, ages)

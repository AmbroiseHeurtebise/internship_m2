import numpy as np
from multiviewica_delay import mvica_ds
from utils import load_and_reduce_data


# parameters
task = "auditory"
n_concat = 6
n_components_pca = 5
n_subjects_subgroup = 50
max_dilation = 1.15
max_shift = 0.05
W_scale = 200
random_state = 42

# load and reduce data
X, subjects, ages, n_subjects_data = load_and_reduce_data(
    task, n_concat, n_components_pca, n_subjects_subgroup, random_state)

# MVICA_DS
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
)

# save data
if n_subjects_subgroup is None:
    n_subjects_subgroup = n_subjects_data
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_camcan/"
W_save_name = f"W_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
dilations_save_name = f"dilations_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
shifts_save_name = f"shifts_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
Y_save_name = f"Y_{task}_task_{n_subjects_subgroup}_{n_components_pca}_{n_concat}.npy"
np.save(results_dir + W_save_name, W_list)
np.save(results_dir + dilations_save_name, dilations)
np.save(results_dir + shifts_save_name, shifts)
np.save(results_dir + Y_save_name, Y_list)

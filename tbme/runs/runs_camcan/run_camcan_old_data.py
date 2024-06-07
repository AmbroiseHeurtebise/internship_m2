import os
import numpy as np
from sklearn.utils.extmath import randomized_svd
from multiviewica_delay import mvica_ds


def load_data(task, nb_subjects):
    data_dir = "/storage/store2/work/aheurteb/mvicad/tbme/data/camcan_old/"
    # Load dataset
    dataset_name = f"X_{task}_task_mag_{nb_subjects}.npy"
    dataset_path = data_dir + dataset_name
    if os.path.exists(dataset_path):
        X = np.load(dataset_path)
    else:
        raise NameError("Dataset not found.")
    # Save subjects
    subjects_name = f"subjects_{task}_task_{nb_subjects}.npy"
    subjects_path = data_dir + subjects_name
    if os.path.exists(subjects_path):
        subjects = np.load(subjects_path)
    else:
        raise NameError("Subjects not found.")
    # save ages
    ages_name = f"ages_{task}_task_{nb_subjects}.npy"
    ages_path = data_dir + ages_name
    if os.path.exists(ages_path):
        ages = np.load(ages_path)
    else:
        raise NameError("Ages not found.")
    return X, subjects, ages


def pca_reduce_data(X, n_components, random_state=None):
    n_groups, n_features, n_samples = X.shape
    reduced = []
    basis = []
    for i in range(n_groups):
        U_i, S_i, V_i = randomized_svd(
            X[i], n_components=n_components, random_state=random_state)
        reduced.append(S_i.reshape(-1, 1) * V_i)
        basis.append(U_i.T)
    return np.array(basis), np.array(reduced)


# parameters
n_components_pca = 5
nb_subjects_subgroup = 50
max_dilation = 1.15
max_shift = 0.05
W_scale = 200
random_state = 42
# load data
X, subjects, ages = load_data("visual", 477)
# apply PCA
P, X = pca_reduce_data(
    X, n_components=n_components_pca, random_state=random_state,
)
# MVICA_DS
W_list, dilations, shifts, Y_list = mvica_ds(
    X_list=X,
    n_concat=1,
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
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_camcan/"
W_save_name = "W_old_data.npy"
dilations_save_name = "dilations_old_data.npy"
shifts_save_name = "shifts_old_data.npy"
Y_save_name = "Y_old_data.npy"
np.save(results_dir + W_save_name, W_list)
np.save(results_dir + dilations_save_name, dilations)
np.save(results_dir + shifts_save_name, shifts)
np.save(results_dir + Y_save_name, Y_list)

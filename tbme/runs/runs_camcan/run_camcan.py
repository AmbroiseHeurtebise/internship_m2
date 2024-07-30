import os
import numpy as np
from multiviewica_delay import mvica_ds
from multiviewica_delay.multiviewica_shifts._reduce_data import pca_reduce_data


def load_data(task, n_subjects, n_concat):
    data_dir = f"/storage/store2/work/aheurteb/mvicad/tbme/data/camcan/{task}/"
    suffix1 = f"_{task}_task_mag_{n_subjects}_{n_concat}.npy"
    suffix2 = f"_{task}_task_{n_subjects}_{n_concat}.npy"
    # Load dataset
    dataset_name = "X" + suffix1
    dataset_path = data_dir + dataset_name
    print(f"dataset path : {dataset_path}")
    if os.path.exists(dataset_path):
        X = np.load(dataset_path)
    else:
        raise NameError("Dataset not found.")
    # Save subjects
    subjects_name = "subjects" + suffix2
    subjects_path = data_dir + subjects_name
    if os.path.exists(subjects_path):
        subjects = np.load(subjects_path)
    else:
        raise NameError("Subjects not found.")
    # save ages
    ages_name = "ages" + suffix2
    ages_path = data_dir + ages_name
    if os.path.exists(ages_path):
        ages = np.load(ages_path)
    else:
        raise NameError("Ages not found.")
    return X, subjects, ages


# parameters
task = "auditory"
n_concat = 6
n_components_pca = 5
n_subjects_subgroup = 50
max_dilation = 1.15
max_shift = 0.05
W_scale = 200
random_state = 42

# define n_subjects_data
if task == "visual":
    n_subjects_data = 477
elif task == "auditory":
    n_subjects_data = 501

# load data
X, subjects, ages = load_data(task, n_subjects_data, n_concat)
print(f"Dataset shape : {X.shape}")

# eventually reduce the number of subjects
if n_subjects_subgroup < n_subjects_data:
    rng = np.random.RandomState(random_state)
    indices = rng.choice(
        np.arange(n_subjects_data), size=n_subjects_subgroup, replace=False)
    X = X[indices]
    subjects = subjects[indices]
    ages = ages[indices]
print(f"Dataset shape after reducing the number of subjects : {X.shape}")

# apply PCA
P, X = pca_reduce_data(
    X, n_components=n_components_pca, random_state=random_state,
)
print(f"Dataset shape after PCA : {X.shape}")

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

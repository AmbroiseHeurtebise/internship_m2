import os
import numpy as np
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


def reduce_nb_subjects(
    X, subjects, ages, n_subjects_subgroup=None, random_state=None,
):
    n_subjects_data = len(X)
    # eventually reduce the number of subjects
    if n_subjects_subgroup is not None:
        assert n_subjects_subgroup <= n_subjects_data
        rng = np.random.RandomState(random_state)
        indices = rng.choice(
            np.arange(n_subjects_data), size=n_subjects_subgroup, replace=False)
        X = X[indices]
        subjects = subjects[indices]
        ages = ages[indices]
    return X, subjects, ages


def load_and_reduce_data(
    task, n_concat, n_components_pca, n_subjects_subgroup=None, random_state=None,
):
    # define n_subjects_data
    if task == "visual":
        n_subjects_data = 477
    elif task == "auditory":
        n_subjects_data = 501
    # load data
    X, subjects, ages = load_data(
        task=task, n_subjects=n_subjects_data, n_concat=n_concat)
    print(f"Dataset shape : {X.shape}")
    # reduce the number of subjects
    X, subjects, ages = reduce_nb_subjects(
        X, subjects, ages, n_subjects_subgroup=n_subjects_subgroup,
        random_state=random_state)
    print(f"Dataset shape after reducing the number of subjects : {X.shape}")
    # PCA
    P, X = pca_reduce_data(
        X, n_components=n_components_pca, random_state=random_state)
    print(f"Dataset shape after PCA : {X.shape}")
    return X, subjects, ages, n_subjects_data

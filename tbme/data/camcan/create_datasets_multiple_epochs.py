import pandas as pd
import numpy as np
import os
import mne
from joblib import Parallel, delayed


# Get list of subjects' names
def get_subjects_names():
    data_passive_path = "/storage/store/data/camcan/BIDSsep/passive"
    # DataFrame containing names of subjects
    participants_path = data_passive_path + "/participants.tsv"
    participants_df = pd.read_csv(participants_path, delimiter="\t")
    # Drop 9 duplicates
    participants_df.drop_duplicates(subset="participant_id", keep="first", inplace=True)
    all_subjects = participants_df["participant_id"].astype('string')
    # Keep only available subjects (618 out of 645)
    available_subjects = []
    derivatives_passive_path = "/storage/store/derivatives/camcan/BIDSsep/passive"
    for subject in all_subjects:
        subject_path = os.path.join(
            derivatives_passive_path, subject, "ses-passive/meg",
            subject + '_ses-passive_task-passive_proc-clean_epo.fif')
        if os.path.isfile(subject_path):
            available_subjects.append(subject)
    return available_subjects, participants_df


# get epochs of one subject
def get_subject_data(subject, task, n_concat=1):
    derivatives_passive_path = "/storage/store/derivatives/camcan/BIDSsep/passive"
    subject_path = os.path.join(
        derivatives_passive_path, subject, "ses-passive/meg",
        subject + '_ses-passive_task-passive_proc-clean_epo.fif')
    # read all epochs
    epochs = mne.read_epochs(subject_path, verbose=False).pick("mag")[task]
    epochs_data = epochs.get_data()
    # average epochs
    n_epochs = len(epochs_data)
    batch_size = n_epochs // n_concat
    epochs_avg = []
    for i in range(n_concat-1):
        epochs_avg.append(np.mean(epochs_data[i*batch_size: (i+1)*batch_size], axis=0))
    epochs_avg.append(np.mean(epochs_data[(n_concat-1)*batch_size:], axis=0))
    return np.array(epochs_avg)


# create a dataset of epochs
def create_dataset(subjects, task, n_concat=1, n_subjects=None, N_JOBS=10):
    if n_subjects is None:
        n_subjects = len(subjects)
    if task == "visual":
        task = "vis"
    elif task == "auditory":
        task = "audio"
    else:
        raise ValueError("Wrong task name")
    X = Parallel(n_jobs=N_JOBS)(
        delayed(get_subject_data)(subject, task, n_concat)
        for subject in subjects[:n_subjects])
    return np.array(X)


# remove some subjects
def remove_subjects(X, subjects, task, n_subjects=None):
    if task == "visual":
        removed_sub = np.array([
            18, 34, 41, 48, 64, 69, 79, 110, 111, 114, 120, 122, 126, 133, 134, 136,
            137, 140, 148, 149, 159, 160, 168, 170, 173, 176, 178, 191, 194, 197, 209,
            211, 216, 217, 226, 229, 240, 243, 249, 269, 270, 271, 276, 277, 283, 287,
            295, 296, 298, 303, 314, 316, 319, 323, 325, 334, 335, 336, 339, 346, 350,
            351, 354, 357, 358, 359, 362, 363, 367, 370, 372, 374, 375, 376, 382, 384,
            388, 390, 391, 392, 393, 397, 400, 408, 413, 422, 423, 437, 438, 441, 443,
            445, 450, 451, 455, 458, 459, 462, 464, 466, 479, 480, 481, 482, 493, 500,
            501, 504, 506, 511, 514, 515, 516, 526, 527, 534, 538, 539, 545, 550, 552,
            554, 559, 562, 563, 569, 571, 574, 578, 579, 580, 581, 585, 590, 596, 600,
            603, 605, 609, 610, 612])
    elif task == "auditory":
        removed_sub = np.array([
            0, 1, 7, 8, 13, 16, 18, 20, 31, 46, 47, 48, 52, 58, 61, 67, 77, 79, 83, 97,
            100, 113, 123, 127, 133, 134, 136, 156, 160, 169, 173, 176, 177, 184, 185,
            209, 213, 229, 233, 235, 239, 243, 249, 255, 257, 262, 270, 271, 276, 283,
            296, 303, 313, 314, 323, 336, 339, 351, 358, 359, 363, 367, 368, 370, 372,
            374, 388, 392, 397, 400, 416, 422, 423, 437, 442, 445, 447, 450, 451, 455,
            457, 458, 462, 464, 468, 469, 474, 479, 482, 494, 500, 501, 502, 504, 506,
            508, 511, 514, 515, 516, 525, 527, 538, 539, 544, 550, 559, 562, 574, 578,
            579, 581, 590, 599, 603, 605, 609])
    else:
        raise ValueError("Wrong task name")
    if n_subjects is not None:
        # XXX not perfect because it produces a dataset of length <= n_subjects
        removed_sub = removed_sub[removed_sub < n_subjects]
    subjects_clean = np.delete(subjects, removed_sub)
    X_clean = np.delete(X, removed_sub, axis=0)
    return X_clean, subjects_clean


# Get ages of the 477 or 501 subjects
def get_subjects_ages(subjects, participants_df):
    age = []
    for subject in subjects:
        assert participants_df["participant_id"].isin([subject]).sum() == 1
        line = participants_df[participants_df["participant_id"].isin([subject])]
        age.append(line["age"].values[0])
    return np.array(age)


# save dataset X, subjects and ages
def save_data(X, subjects, ages, task, n_concat=1):
    data_dir = f"/storage/store2/work/aheurteb/mvicad/tbme/data/camcan/{task}/"
    # Save dataset
    dataset_name = f"X_{task}_task_mag_{len(X)}_{n_concat}.npy"
    dataset_path = data_dir + dataset_name
    if os.path.exists(dataset_path):
        print("Dataset already exists.")
    else:
        np.save(dataset_path, X)
    # Save subjects
    subjects_name = f"subjects_{task}_task_{len(X)}_{n_concat}.npy"
    subjects_path = data_dir + subjects_name
    if os.path.exists(subjects_path):
        print("Subjects info already exist.")
    else:
        np.save(subjects_path, subjects)
    # save ages
    ages_name = f"ages_{task}_task_{len(X)}_{n_concat}.npy"
    ages_path = data_dir + ages_name
    if os.path.exists(ages_path):
        print("Ages info already exist.")
    else:
        np.save(ages_path, ages)


# parameters
task = "visual"
n_concat = 3
n_subjects = None
N_JOBS = 4

# create and save data
subjects, all_participants_df = get_subjects_names()
X = create_dataset(
    subjects=subjects, task=task, n_concat=n_concat, n_subjects=n_subjects, N_JOBS=N_JOBS)
X_clean, subjects_clean = remove_subjects(X=X, subjects=subjects, task=task, n_subjects=n_subjects)
ages = get_subjects_ages(subjects=subjects_clean, participants_df=all_participants_df)
save_data(X=X_clean, subjects=subjects_clean, ages=ages, task=task, n_concat=n_concat)

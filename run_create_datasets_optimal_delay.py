import pandas as pd
import numpy as np
import os
import mne
from multiviewica import multiviewica
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle


def compute_evoked_one_subject(patient_id, dataset_passive_path):
    patient_path = os.path.join(dataset_passive_path, patient_id, "ses-passive", "meg", patient_id + '_ses-passive_task-passive_proc-clean_epo.fif')
    epochs = mne.read_epochs(patient_path)
    mag_picks = mne.pick_types(epochs.info, meg='mag')
    # We arbitrarily chose the stimulus 'audio/600Hz'
    evoked = epochs['audio/600Hz'].average().get_data()[mag_picks]
    return evoked


def compute_argmax_one_subject(patient_id, dataset_passive_path):
    evoked = compute_evoked_one_subject(patient_id, dataset_passive_path)
    norm_evoked = np.linalg.norm(evoked, axis=0)
    ind = np.argmax(norm_evoked)
    return ind


def compute_argmax_total(dataset_passive_path, N_JOBS):
    savefile_name = "argmax_evoked_camcan.pkl"

    if os.path.isfile(savefile_name):
        with open(savefile_name, "rb") as save_argmax_file:
            list_patient_id, argmax_subjects = pickle.load(save_argmax_file)

    else:
        list_patient_id = []
        for patient_id in os.listdir(dataset_passive_path):
            if("sub-CC" in patient_id):
                list_patient_id.append(patient_id)

        argmax_subjects = Parallel(n_jobs=N_JOBS)(delayed(compute_argmax_one_subject)(id, dataset_passive_path) for id in tqdm(list_patient_id))

        with open(savefile_name, "wb") as save_argmax_file:
            pickle.dump([list_patient_id, argmax_subjects], save_argmax_file)

    return list_patient_id, argmax_subjects


def get_cohorts(dataset_passive_path, N_JOBS):
    # Compute the argmax of the norm of the evoked response of the subjects 
    list_patient_id, argmax_subjects = compute_argmax_total(dataset_passive_path, N_JOBS)
    argmax_median = np.median(argmax_subjects)

    # Create dataframe
    data = {'participant_id': list_patient_id, 'argmax': argmax_subjects}
    df = pd.DataFrame(data)
    which_group = [(argmax > argmax_median) * 1 for argmax in argmax_subjects]
    df['group'] = which_group

    # Create 2 cohorts of subjects
    cohort1 = df[df['group'] == 0]['participant_id']
    cohort2 = df[df['group'] == 1]['participant_id']
    return cohort1, cohort2


def get_datasets(cohort1, cohort2, dataset_passive_path, N_JOBS):
    savefile_name = "datasets_by_argmax.pkl"

    if os.path.isfile(savefile_name):
        with open(savefile_name, "rb") as save_datasets_file:
            dataset1, dataset2 = pickle.load(save_datasets_file)

    else:
        dataset1 = Parallel(n_jobs=N_JOBS)(delayed(
            compute_evoked_one_subject)(id, dataset_passive_path) for id in tqdm(cohort1))
        dataset2 = Parallel(n_jobs=N_JOBS)(delayed(
            compute_evoked_one_subject)(id, dataset_passive_path) for id in tqdm(cohort2))
        dataset1 = np.asarray(dataset1)
        dataset2 = np.asarray(dataset2)

        with open(savefile_name, "wb") as save_datasets_file:
            pickle.dump([dataset1, dataset2], save_datasets_file)

    return dataset1, dataset2


def mvica(dataset1, dataset2, perform_ica=0):
    savefile_name = "mvica_datasets.pkl"

    if not os.path.isfile(savefile_name) or perform_ica:
        _, _, S1 = multiviewica(dataset1)
        _, _, S2 = multiviewica(dataset2)

        with open(savefile_name, "wb") as save_ica_file:
            pickle.dump([S1, S2], save_ica_file)


if __name__ == '__main__':
    # Path
    dataset_passive_path = "/storage/store/derivatives/camcan/BIDSsep/passive"

    # Dataframe containing the age of the subjects
    # participants_path = "/storage/store/data/camcan/BIDSsep/passive/participants.tsv"
    # participants_df = pd.read_csv(participants_path, delimiter="\t")

    # Get 2 cohorts of subject
    N_JOBS = 10
    cohort1, cohort2 = get_cohorts(dataset_passive_path, N_JOBS)

    # Get a dataset for each cohort
    dataset1, dataset2 = get_datasets(cohort1, cohort2, dataset_passive_path, N_JOBS)

    # Perform MVICA
    perform_ica = 0  # wether we perform again ICA or we just load the precedent results
    mvica(dataset1, dataset2, perform_ica=perform_ica)

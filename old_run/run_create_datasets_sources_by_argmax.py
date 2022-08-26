import pandas as pd
import numpy as np
import os
import mne
from multiviewica import multiviewica
from joblib import Parallel, delayed
from tqdm import tqdm
from joblib import Memory


def compute_evoked_one_subject(participant_id, dataset_passive_path):
    patient_path = os.path.join(
        dataset_passive_path,
        participant_id,
        "ses-passive",
        "meg",
        participant_id + "_ses-passive_task-passive_proc-clean_epo.fif",
    )
    epochs = mne.read_epochs(patient_path)
    # We arbitrarily chose the stimulus 'audio/600Hz'
    evoked = epochs["audio/600Hz"].average()
    return evoked.pick("mag")


def compute_argmax_one_subject(participant_id, dataset_passive_path):
    evoked = compute_evoked_one_subject(participant_id, dataset_passive_path)
    norm_evoked = np.linalg.norm(evoked.data, axis=0)
    ind = np.argmax(norm_evoked)
    return dict(
        ind=ind,
        time=evoked.times[ind],
        value=norm_evoked[ind],
        participant_id=participant_id,
    )


def compute_argmax_total(dataset_passive_path, list_participants_id, N_JOBS):
    argmax_subjects = Parallel(n_jobs=N_JOBS)(
        delayed(compute_argmax_one_subject)(id, dataset_passive_path)
        for id in tqdm(list_participants_id)
    )

    argmax_subjects = pd.DataFrame(argmax_subjects)
    # argmax_subjects.to_csv("data/peak_latencies.csv")

    return argmax_subjects


def get_cohorts(dataset_passive_path, list_participants_id, N_JOBS, n_participants):
    list_participants_id = list_participants_id[:n_participants]

    # Compute the argmax of the norm of the evoked response of the subjects
    argmax_subjects = compute_argmax_total(dataset_passive_path, list_participants_id, N_JOBS)
    # argmax_subjects = argmax_subjects.set_index("participant_id")

    # Remove participants with argmax < 200 or > 400
    argmax_subjects = argmax_subjects[(argmax_subjects["ind"] >= 200) & (argmax_subjects["ind"] <= 400)]

    # Remove other participants
    remove_ind = [2, 10, 11, 23, 30, 31, 32, 33, 35, 38, 40, 44, 54, 59, 65, 75, 77, 79, 85, 86, 88, 90, 93, 94, 95, 96, 100, 102, 112, 118, 120, 123, 125, 131, 143, 154, 156, 160, 162, 166, 168, 174, 175, 177, 185, 187, 188, 190, 195, 199, 202, 209, 212, 213, 215, 218, 219, 231, 232, 233, 237, 239, 255, 260, 266, 268, 273, 274, 276, 277, 278, 282, 285, 293, 294, 298, 299, 300, 310, 315, 322, 323, 329, 330, 333, 335, 338, 339, 344, 348, 352, 356, 358, 359, 366, 367, 369, 371, 375, 376, 387, 390, 393, 400, 409, 419, 424, 430, 432, 440, 448, 450, 451, 457, 464, 476, 484, 485, 488, 489, 490, 494, 500, 506, 509, 510, 511, 519, 523, 524, 529, 532, 535, 546, 551, 552, 555, 558, 560, 563, 568, 570, 573, 580, 581, 589, 590, 591, 593, 595, 606, 608, 614, 615, 616]
    argmax_subjects.drop(remove_ind, inplace=True)

    # Select first and last quarter of participants, wrt the argmax 
    # argmax_median = argmax_subjects.median()
    argmax_Q1 = argmax_subjects.quantile(0.25)
    argmax_Q3 = argmax_subjects.quantile(0.75)

    # Create cohorts
    cohort1 = argmax_subjects[argmax_subjects["ind"] <= argmax_Q1["ind"]]
    cohort2 = argmax_subjects[argmax_subjects["ind"] > argmax_Q3["ind"]]

    return cohort1, cohort2


def get_datasets(cohort1, cohort2, dataset_passive_path, N_JOBS):
    savefile_name1 = "data/dataset1.npy"
    savefile_name2 = "data/dataset2.npy"

    if os.path.isfile(savefile_name1) and os.path.isfile(savefile_name2):
        with open(savefile_name1, "rb") as save_dataset1_file:
            dataset1 = np.load(save_dataset1_file)
        with open(savefile_name2, "rb") as save_dataset2_file:
            dataset2 = np.load(save_dataset2_file)

    else:
        dataset1 = Parallel(n_jobs=N_JOBS)(
            delayed(compute_evoked_one_subject)(id, dataset_passive_path)
            for id in tqdm(cohort1["participant_id"])
        )
        dataset2 = Parallel(n_jobs=N_JOBS)(
            delayed(compute_evoked_one_subject)(id, dataset_passive_path)
            for id in tqdm(cohort2["participant_id"])
        )
        dataset1 = np.array([d.data for d in dataset1])
        dataset2 = np.array([d.data for d in dataset2])

        np.save(savefile_name1, dataset1)
        np.save(savefile_name2, dataset2)

    return dataset1, dataset2


def mvica(dataset1, dataset2, n_components):
    _, _, S1 = multiviewica(dataset1, n_components=n_components)
    _, _, S2 = multiviewica(dataset2, n_components=n_components)
    return S1, S2


if __name__ == "__main__":
    # Path
    dataset_passive_path = "/storage/store/derivatives/camcan/BIDSsep/passive"

    # Participants of the derivatives folder
    list_participants_id = []
    for participant_id in os.listdir(dataset_passive_path):
        if "sub-CC" in participant_id:
            list_participants_id.append(participant_id)
    n_participants = len(list_participants_id)  # warning: some particpants will be removed

    mem = Memory(".")
    # Get 2 cohorts of subject
    N_JOBS = 5
    cohort1, cohort2 = mem.cache(get_cohorts)(
        dataset_passive_path, list_participants_id, N_JOBS, n_participants
    )

    print("Size of cohort 1 : {}".format(len(cohort1)))
    print("Size of cohort 2 : {}".format(len(cohort2)))

    # Get a dataset for each cohort
    dataset1, dataset2 = mem.cache(get_datasets)(
        cohort1, cohort2, dataset_passive_path, N_JOBS
    )

    # Perform MVICA
    n_components = 6
    S1, S2 = mem.cache(mvica)(dataset1, dataset2, n_components=n_components)
    np.save("data/sources1.npy", S1)
    np.save("data/sources2.npy", S2)

import glob

import numpy as np
import mne

from joblib import Parallel, delayed
from mne.minimum_norm import make_inverse_operator, apply_inverse
from params import setup, expe_type

def do_expe(folder_path, algo):
    n_concat = setup['n_concat']
    n_subjects = setup['n_subjects']
    audio = setup['audio']
    n_comp = setup['n_comp']
    task = setup['task']
    task_str = {True: 'task', False: 'passive'}[task]

    save_str = '%d_%d_%d_%s_%s_0_%s' % (n_subjects, n_comp, n_concat, expe_type, task_str, algo)
    subjects = np.load(folder_path + 'subjects_%s.npy' % save_str)
    # save_str = '%d_%d_%d_%s_%s_%s' % (n_subjects, n_comp, n_concat, expe_type, task_str, algo)
    mixing = np.load(folder_path + 'mixing_%s.npy' % save_str)
    subjects_dir = '/storage/store/data/camcan-mne/freesurfer/'
    bads = []
    adr = '/storage/store2/work/rhochenb/Data/Cam-CAN/BIDS/derivatives/mne-study-template/sub-CC%s/meg/sub-CC%s_task-%s%s.fif'
    stcs = []

    def compute(i, subject):
        try:
            fname = adr % (subject, subject, task_str, '_cleaned-epo')
            fwd_name = adr % (subject, subject, task_str, '-fwd')
            fwd = mne.read_forward_solution(fwd_name)
            fwd = mne.pick_types_forward(fwd, meg='mag', eeg=False)
            evoked = mne.read_epochs(fname, proj=True)
            info = evoked.info
            picks = mne.pick_types(info, meg='mag')
            info_ = mne.pick_info(info, picks)
            cov = mne.make_ad_hoc_cov(info_, std=None, verbose=None)
            inv = make_inverse_operator(info_, fwd, cov)
            evoked = mne.EvokedArray(mixing[i], info_)
            # stc = apply_inverse(evoked, inv, method='eLORETA')
            stc = apply_inverse(evoked, inv, method='sLORETA', lambda2=1/64)
            subject = 'CC' + str(subject)

            morph = mne.compute_source_morph(stc, subject_from=subject,
                                            subject_to='fsaverage',
                                            subjects_dir=subjects_dir)
            stc_fsaverage = morph.apply(stc)
            save_str = '%d_%d_%d_%s_%s_%i' % (len(subjects), n_comp, n_concat, expe_type, task_str, i)
            stc_fsaverage.save(folder_path + 'stc_%s' % save_str)
        except:
            print(i)
            return None
        return stc_fsaverage
        #except:
        #    bads.append(subject)

    stcs = Parallel(n_jobs=30)(delayed(compute)(i, subject) for i, subject in enumerate(subjects))
    goods = []
    bads = []
    for i, stc in enumerate(stcs):
        if stc is None:
            bads.append(subjects[i])
        else:
            goods.append(stc)

    stc_avg = sum(goods) / len(goods)
    save_str = '%d_%d_%d_%s_%s' % (len(subjects), n_comp, n_concat, expe_type, task_str)
    stc_avg.save(folder_path + 'stc_avg_%s' % save_str)

# do_expe("/storage/store2/work/hrichard/amvica/", "amvica")
# do_expe("/storage/store2/work/hrichard/extendedcanica/", "extendedcanica")
do_expe("/storage/store2/work/hrichard/canica/", "canica")

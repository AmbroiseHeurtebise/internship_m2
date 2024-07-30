import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from utils_runs import run_experiment


def run_experiment_wrapped(varying_output, **kwargs):
    max_dilation, max_shift = varying_output
    dict_varying_outputs = {
        "max_dilation": max_dilation,
        "max_shift": max_shift,
    }
    dict_expe = run_experiment(dict_varying_outputs=dict_varying_outputs, **kwargs)
    return dict_expe


# fixed params
N_JOBS = 8
m = 5
p = 3
n_concat = 5
n = 600
noise_data = 0.01
noise_model = 1
n_bins = 10
freq_level = 50
S1_S2_scale = 0.7
number_of_filters_squarenorm_f = 1
filter_length_squarenorm_f = 2
use_envelop_term = True
number_of_filters_envelop = 1
filter_length_envelop = 5
dilation_scale_per_source = True
W_scale = 20
penalization_scale = 1
nb_points_grid_init = 10
verbose = False
return_all_iterations = True

# varying params
nb_seeds = 30
random_states = np.arange(nb_seeds)
eps = 1e-5
max_dilation_all = np.concatenate([np.array([1+eps]), np.linspace(1.03, 1.3, 10)])
max_shift_all = np.concatenate([np.array([eps]), np.linspace(0.01, 0.1, 10)])
max_dilation_max_shift_pairs = list(zip(max_dilation_all, max_shift_all))
nb_expes = len(max_dilation_all) * len(random_states)

# run experiment
print(f"\nTotal number of experiments : {nb_expes}")
print("\n############################################### Start ###############################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment_wrapped)(
        varying_output=(max_dilation, max_shift),
        m=m,
        p=p,
        n_concat=n_concat,
        n=n,
        max_dilation=max_dilation,
        max_shift=max_shift,
        noise_data=noise_data,
        noise_model=noise_model,
        n_bins=n_bins,
        freq_level=freq_level,
        S1_S2_scale=S1_S2_scale,
        number_of_filters_squarenorm_f=number_of_filters_squarenorm_f,
        filter_length_squarenorm_f=filter_length_squarenorm_f,
        use_envelop_term=use_envelop_term,
        number_of_filters_envelop=number_of_filters_envelop,
        filter_length_envelop=filter_length_envelop,
        dilation_scale_per_source=dilation_scale_per_source,
        W_scale=W_scale,
        penalization_scale=penalization_scale,
        random_state=random_state,
        nb_points_grid_init=nb_points_grid_init,
        verbose=verbose,
        return_all_iterations=return_all_iterations,
    ) for (max_dilation, max_shift), random_state
    in product(max_dilation_max_shift_pairs, random_states)
)
print("\n######################################### Obtained DataFrame #########################################")
df_res = pd.DataFrame(dict_res)
print(df_res)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_max_dilation_and_max_shift"
save_path = results_dir + save_name
df_res.to_csv(save_path, index=False)
print("\n################################################ End ################################################")

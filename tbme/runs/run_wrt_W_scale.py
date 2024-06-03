import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from utils_runs import run_experiment


def run_experiment_wrapped(varying_output, **kwargs):
    dict_varying_outputs = {"W_scale": varying_output}
    dict_expe = run_experiment(dict_varying_outputs=dict_varying_outputs, **kwargs)
    return dict_expe


# fixed params
N_JOBS = 4
m = 5
p = 3
n_concat = 5
n = 600
max_shift = 0.05
max_dilation = 1.15
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
penalization_scale = 1e-5
nb_points_grid_init = 10
verbose = False
return_all_iterations = True

# varying params
nb_seeds = 30
random_states = np.arange(nb_seeds)
W_scales = 2. ** np.arange(-2, 6)  # [0.25, 0.5, 1., 2., 4., 8., 16., 32.]
nb_expes = len(W_scales) * len(random_states)

# run experiment
print(f"\nTotal number of experiments : {nb_expes}")
print("\n############################################### Start ###############################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment_wrapped)(
        varying_output=W_scale,
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
    ) for W_scale, random_state
    in product(W_scales, random_states)
)
print("\n######################################### Obtained DataFrame #########################################")
df_res = pd.DataFrame(dict_res)
print(df_res)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_W_scale"
save_path = results_dir + save_name
df_res.to_csv(save_path, index=False)
print("\n################################################ End ################################################")

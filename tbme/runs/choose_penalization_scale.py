import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from utils_runs import run_experiment


# fixed params
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
number_of_filters_squarenorm_f = 0
filter_length_squarenorm_f = 3
use_envelop_term = True
number_of_filters_envelop = 1
filter_length_envelop = 10
dilation_scale_per_source = True
W_scale = 15
nb_points_grid_init = 10
verbose = False
return_all_iterations = True

# varying params
penalization_scales = np.logspace(-5, 0, 11)
nb_seeds = 30
random_states = np.arange(nb_seeds)
nb_expes = len(penalization_scales) * len(random_states)

# run experiment
print("\n############################################### Start ###############################################")
df_res = pd.DataFrame()
for i, (penalization_scale, random_state) in tqdm(enumerate(product(penalization_scales, random_states))):
    dict_varying_outputs = {
        "penalization scale": penalization_scale,
    }
    dict_expe = run_experiment(
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
        dict_varying_outputs=dict_varying_outputs,
    )
    df = pd.DataFrame(dict_expe, index=[i])
    df_res = pd.concat([df_res, df], ignore_index=True)
    print(f"Total number of experiments : {nb_expes}\n")
print("\n######################################### Obtained DataFrame #########################################")
print(df_res)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_penalization_scale"
save_path = results_dir + save_name
df_res.to_csv(save_path, index=False)
print("\n################################################ End ################################################")

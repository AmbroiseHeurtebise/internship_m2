import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from utils_runs import run_experiment


def run_experiment_wrapped(varying_output_row, **kwargs):
    dict_varying_outputs = varying_output_row.to_dict()
    del dict_varying_outputs["random_state"]
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
dilation_scale_per_source = True
W_scale = 15
penalization_scale = 1e-5
nb_points_grid_init = 10
verbose = False
return_all_iterations = True

# varying params
nb_seeds = 2
random_states = np.arange(nb_seeds)
filter_length_squarenorm_f_all = np.array([1, 2, 3, 5, 7, 10, 15])
number_of_filters_envelop_all = np.array([1, 2])
filter_length_envelop_all = np.array([2, 3, 5, 10, 15, 20, 25])

# DataFrame of combinations
columns = [
    "filter_length_squarenorm_f", "use_envelop_term", "number_of_filters_envelop",
    "filter_length_envelop", "random_state"]
df_varying_outputs = pd.DataFrame(columns=columns)

# first 98*nb_seeds combinations
for i, (
    filter_length_squarenorm_f, number_of_filters_envelop, filter_length_envelop,
    random_state) in enumerate(
        product(filter_length_squarenorm_f_all, number_of_filters_envelop_all,
                filter_length_envelop_all, random_states)):
    row = {
        "filter_length_squarenorm_f": filter_length_squarenorm_f,
        "use_envelop_term": True,
        "number_of_filters_envelop": number_of_filters_envelop,
        "filter_length_envelop": filter_length_envelop,
        "random_state": random_state,
    }
    df_varying_outputs.loc[i] = row

# 7*nb_seeds following combinations
for j, (
    filter_length_squarenorm_f, random_state) in enumerate(
        product(filter_length_squarenorm_f_all, random_states)):
    row = {
        "filter_length_squarenorm_f": filter_length_squarenorm_f,
        "use_envelop_term": True,
        "number_of_filters_envelop": 1,
        "filter_length_envelop": 1,
        "random_state": random_state,
    }
    df_varying_outputs.loc[i+j+1] = row

# 7*nb_seeds last combinations
for k, (
    filter_length_squarenorm_f, random_state) in enumerate(
        product(filter_length_squarenorm_f_all, random_states)):
    row = {
        "filter_length_squarenorm_f": filter_length_squarenorm_f,
        "use_envelop_term": False,
        "number_of_filters_envelop": 0,
        "filter_length_envelop": 0,
        "random_state": random_state,
    }
    df_varying_outputs.loc[i+j+k+2] = row

nb_expes = len(df_varying_outputs)

# run experiment
print(f"\nTotal number of experiments : {nb_expes}")
print("\n############################################### Start ###############################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment_wrapped)(
        varying_output_row=row,
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
        filter_length_squarenorm_f=row["filter_length_squarenorm_f"],
        use_envelop_term=row["use_envelop_term"],
        number_of_filters_envelop=row["number_of_filters_envelop"],
        filter_length_envelop=row["filter_length_envelop"],
        dilation_scale_per_source=dilation_scale_per_source,
        W_scale=W_scale,
        penalization_scale=penalization_scale,
        random_state=row["random_state"],
        nb_points_grid_init=nb_points_grid_init,
        verbose=verbose,
        return_all_iterations=return_all_iterations,
    ) for _, row in df_varying_outputs.iterrows()
)
print("\n######################################### Obtained DataFrame #########################################")
df_res = pd.DataFrame(dict_res)
print(df_res)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_different_filters_combinations"
save_path = results_dir + save_name
df_res.to_csv(save_path, index=False)
print("\n################################################ End ################################################")

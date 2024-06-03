import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# read dataframe
nb_seeds = 20
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_penalization_scale"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# plot Amari distance
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df, x="penalization_scale", y="Amari MVICAD", linewidth=2.5,
    label="MVICAD", estimator=np.median, c=colors[1])
sns.lineplot(
    data=df, x="penalization_scale", y="Amari MVICAD ext", linewidth=2.5,
    label="MVICAD extended", estimator=np.median, c=colors[2])
sns.lineplot(
    data=df, x="penalization_scale", y="Amari LBFGSB", linewidth=2.5,
    label="LBFGSB", estimator=np.median, c=colors[0])
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Amari distance")
plt.grid()
plt.title("Median Amari distance with respect to penalization scale")
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_penalization_scale.pdf")
plt.show()

# plot shift and dilation's errors
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df, x="penalization_scale", y="Dilations error LBFGSB", linewidth=2.5,
    label="dilations error", estimator=np.median)
sns.lineplot(
    data=df, x="penalization_scale", y="Shifts error LBFGSB", linewidth=2.5,
    label="shifts error", estimator=np.median)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Error")
plt.grid()
plt.title("Dilation and shift errors wrt penalization scale")
plt.savefig(figures_dir + "dilation_shift_errors_wrt_penalization_scale.pdf")
plt.show()

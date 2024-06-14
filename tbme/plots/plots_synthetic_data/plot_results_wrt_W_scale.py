import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# read dataframe
nb_seeds = 30
m = 5
p = 3
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_W_scale_m{m}_p{p}"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# plot Amari distance
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df, x="W_scale", y="Amari MVICAD", linewidth=2.5,
    label="MVICAD", estimator=np.median, c=colors[1])
sns.lineplot(
    data=df, x="W_scale", y="Amari MVICAD ext", linewidth=2.5,
    label="MVICAD extended", estimator=np.median, c=colors[2])
sns.lineplot(
    data=df, x="W_scale", y="Amari permica", linewidth=2.5,
    label="PermICA", estimator=np.median, c=colors[3])
sns.lineplot(
    data=df, x="W_scale", y="Amari LBFGSB", linewidth=2.5,
    label="LBFGSB", estimator=np.median, c=colors[0])
plt.legend()
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("W scale")
plt.ylabel("Amari distance")
plt.grid()
plt.title(f"Median Amari distance with respect to W scale ; m={m} and p={p}")
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_W_scale.pdf")
plt.show()

# plot shift and dilation's errors
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df, x="W_scale", y="Dilations error LBFGSB", linewidth=2.5,
    label="dilations error", estimator=np.median, c=colors[0], linestyle=":",
    marker="^")
sns.lineplot(
    data=df, x="W_scale", y="Shifts error LBFGSB", linewidth=2.5,
    label="shifts error", estimator=np.median, c=colors[0], linestyle="--",
    marker="o")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("W scale")
plt.ylabel("Error")
plt.grid()
plt.title(f"Dilation and shift errors wrt W scale ; m={m} and p={p}")
# plt.subplots_adjust(bottom=0.2)
plt.savefig(figures_dir + "dilation_shift_errors_wrt_W_scale.pdf")
plt.show()

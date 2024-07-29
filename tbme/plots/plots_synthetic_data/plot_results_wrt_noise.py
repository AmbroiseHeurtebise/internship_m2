import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_noise"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# plot Amari distance
fontsize = 12
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
sns.lineplot(
    data=df, x="noise_data", y="Amari GroupICA", linewidth=2.5,
    label="GroupICA", estimator=np.median, c=colors[4])
sns.lineplot(
    data=df, x="noise_data", y="Amari MVICA", linewidth=2.5,
    label="MVICA", estimator=np.median, c=colors[2])
sns.lineplot(
    data=df, x="noise_data", y="Amari permica", linewidth=2.5,
    label="PermICA", estimator=np.median, c=colors[3])
sns.lineplot(
    data=df, x="noise_data", y="Amari MVICAD ext", linewidth=2.5,
    label="MVICAD", estimator=np.median, c=colors[1])
sns.lineplot(
    data=df, x="noise_data", y="Amari LBFGSB", linewidth=2.5,
    label="MVICAD$^2$", estimator=np.median, c=colors[0])
plt.legend(fontsize=fontsize)
plt.xscale("log")
plt.yscale("log")
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
plt.xlabel("Noise", fontsize=fontsize)
plt.ylabel("Median Amari distance", fontsize=fontsize)
plt.grid()
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_noise.pdf")
plt.show()

# plot shift and dilation's errors
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df, x="noise_data", y="Dilations error LBFGSB", linewidth=2.5,
    label="dilations error", estimator=np.median, c=colors[0], linestyle=":",
    marker="^")
sns.lineplot(
    data=df, x="noise_data", y="Shifts error LBFGSB", linewidth=2.5,
    label="shifts error", estimator=np.median, c=colors[0], linestyle="--",
    marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Noise")
plt.ylabel("Error")
plt.grid()
plt.title("Dilation and shift errors wrt noise")
plt.savefig(figures_dir + "dilation_shift_errors_wrt_noise.pdf")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# read dataframe
nb_seeds = 20
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_n_subjects"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# plot Amari distance
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

sns.lineplot(
    data=df, x="m", y="Amari MVICAD", linewidth=2.5,
    label="MVICAD", estimator=np.median, c=colors[1])
sns.lineplot(
    data=df, x="m", y="Amari MVICAD ext", linewidth=2.5,
    label="MVICAD extended", estimator=np.median, c=colors[2])
sns.lineplot(
    data=df, x="m", y="Amari permica", linewidth=2.5,
    label="PermICA", estimator=np.median, c=colors[3])
sns.lineplot(
    data=df, x="m", y="Amari LBFGSB", linewidth=2.5,
    label="LBFGSB", estimator=np.median, c=colors[0])
plt.legend()
plt.yscale("log")
plt.xlabel("Number of subjects")
plt.ylabel("Amari distance")
xticks = np.unique(df["m"]).astype(int)
plt.xticks(xticks)
plt.grid()
plt.title("Median Amari distance with respect to the number of subjects")
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_n_subjects.pdf")
plt.show()

# plot shift and dilation's errors
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df, x="m", y="Dilations error LBFGSB", linewidth=2.5,
    label="dilations error", estimator=np.median, linestyle=":",
    marker="^")
sns.lineplot(
    data=df, x="m", y="Shifts error LBFGSB", linewidth=2.5,
    label="shifts error", estimator=np.median, c=colors[0], linestyle="--",
    marker="o")
plt.yscale("log")
plt.xlabel("Number of subjects")
plt.ylabel("Error")
plt.xticks(xticks)
plt.grid()
plt.title("Dilation and shift errors wrt the number of subjects")
plt.savefig(figures_dir + "dilation_shift_errors_wrt_n_subjects.pdf")
plt.show()

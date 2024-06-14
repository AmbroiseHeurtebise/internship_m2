import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_dilation_scale_per_source"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# plot Amari distance
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(figsize=(6, 4))
sns.boxplot(
    data=df, x="dilation_scale_per_source", y="Amari LBFGSB", hue="dilation_scale_per_source",
    linewidth=2.5, palette=colors[:2], legend=False)
plt.yscale("log")
plt.ylabel("Amari distance")
plt.grid()
plt.title("Amari distance for dilation_scale_per_source=[False, True]")
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_dilation_scale_per_source.pdf")
plt.show()

# plot shift and dilation's errors
plt.figure(figsize=(12, 4))
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
sns.boxplot(
    data=df, x="dilation_scale_per_source", y="Dilations error LBFGSB",
    hue="dilation_scale_per_source", linewidth=2.5, palette=colors[:2], legend=False)
plt.yscale("log")
plt.grid()
plt.ylabel("")
plt.title("Dilation error")
plt.subplot(1, 2, 2)
sns.boxplot(
    data=df, x="dilation_scale_per_source", y="Shifts error LBFGSB",
    hue="dilation_scale_per_source", linewidth=2.5, palette=colors[:2], legend=False)
plt.yscale("log")
plt.grid()
plt.ylabel("")
plt.title("Shift error")
plt.suptitle("Dilation and shift errors for dilation_scale_per_source=[False, True]")
plt.savefig(figures_dir + "dilation_shift_errors_wrt_dilation_scale_per_source.pdf")
plt.tight_layout()
plt.show()

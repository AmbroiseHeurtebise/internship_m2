import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_penalization_scale"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# get Times New Roman font
fontsize = 16
font_path = "/storage/store2/work/aheurteb/mvicad/tbme/fonts/Times_New_Roman.ttf"
font_properties = FontProperties(fname=font_path, size=fontsize)

# get colors cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plot Amari distance
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=df, x="penalization_scale", y="Amari LBFGSB", linewidth=2.5,
    label="MVICAD$^2$", estimator=np.median, c=colors[0])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Parameter penalization_scale", font_properties=font_properties)
plt.ylabel("Median Amari distance", font_properties=font_properties)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)
ax.legend_.remove()
plt.grid()
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_penalization_scale.pdf", bbox_inches="tight")
plt.show()

# plot shift and dilation's errors
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=df, x="penalization_scale", y="Dilations error LBFGSB", linewidth=2.5,
    label="dilations error", estimator=np.median, c=colors[0], linestyle=":",
    marker="^")
sns.lineplot(
    data=df, x="penalization_scale", y="Shifts error LBFGSB", linewidth=2.5,
    label="shifts error", estimator=np.median, c=colors[0], linestyle="--",
    marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Parameter penalization_scale", font_properties=font_properties)
plt.ylabel("Error", font_properties=font_properties)
ax.set_yticklabels([], minor=True)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)
plt.legend(prop=font_properties)
plt.grid()
plt.savefig(figures_dir + "dilation_shift_errors_wrt_penalization_scale.pdf", bbox_inches="tight")
plt.show()

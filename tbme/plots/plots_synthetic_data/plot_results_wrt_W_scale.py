import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


def plot_figure_amari(df, ax, font_properties, xlabel=True):
    sns.lineplot(
        data=df, x="W_scale", y="Amari LBFGSB", linewidth=2.5,
        label="MVICAD$^2$", estimator=np.median, c=colors[0])
    plt.xscale("log", base=2)
    plt.yscale("log")
    if xlabel:
        plt.xlabel("Parameter W_scale", font_properties=font_properties)
    else:
        plt.xlabel("")
    plt.ylabel("Median Amari distance", font_properties=font_properties)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax.legend_.remove()
    plt.grid()


def plot_figure_errors(df, ax, font_properties, xlabel=True):
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
    if xlabel:
        plt.xlabel("Parameter W_scale", font_properties=font_properties)
    else:
        plt.xlabel("")
    plt.ylabel("Error", font_properties=font_properties)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    plt.legend(prop=font_properties)
    plt.grid()


# read dataframe
nb_seeds = 30
m = 5
p = 3
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_W_scale_m{m}_p{p}"
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
plot_figure_amari(df, ax, font_properties)
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_W_scale.pdf", bbox_inches="tight")

# plot shift and dilation's errors
fig, ax = plt.subplots(figsize=(6, 4))
plot_figure_errors(df, ax, font_properties)
plt.savefig(figures_dir + "dilation_shift_errors_wrt_W_scale.pdf", bbox_inches="tight")

# plot both figures side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_figure_amari(df, axes[0], font_properties, xlabel=False)
plt.subplot(1, 2, 2)
plot_figure_errors(df, axes[1], font_properties, xlabel=False)
fig.supxlabel("Parameter W_scale", fontsize=fontsize, font_properties=font_properties)
plt.tight_layout(pad=0.5, w_pad=4.0)
plt.savefig(figures_dir + "both_figures_wrt_W_scale.pdf", bbox_inches="tight")

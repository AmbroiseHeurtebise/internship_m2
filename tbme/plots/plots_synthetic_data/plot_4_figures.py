import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


def plot_figure_amari(
    df, x, xlabel, ax, font_properties=None, xscale_log=False, fig=None,
    change_xticks=None,
):
    sns.lineplot(
        data=df, x=x, y="Amari GroupICA", linewidth=2.5,
        label="GroupICA", estimator=np.median, c=colors[4], ax=ax)
    sns.lineplot(
        data=df, x=x, y="Amari MVICA", linewidth=2.5,
        label="MVICA", estimator=np.median, c=colors[2], ax=ax)
    sns.lineplot(
        data=df, x=x, y="Amari permica", linewidth=2.5,
        label="PermICA", estimator=np.median, c=colors[3], ax=ax)
    sns.lineplot(
        data=df, x=x, y="Amari MVICAD ext", linewidth=2.5,
        label="MVICAD", estimator=np.median, c=colors[1], ax=ax)
    sns.lineplot(
        data=df, x=x, y="Amari LBFGSB", linewidth=2.5,
        label="MVICAD$^2$", estimator=np.median, c=colors[0], ax=ax)
    ax.set_xlabel(xlabel, font_properties=font_properties)
    ax.set_ylabel("")
    if xscale_log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    if change_xticks is not None:
        xticks = np.unique(df[x]).astype(int)
        if change_xticks == "half":
            xticks = xticks[2 * np.arange((len(xticks) + 1) // 2)]
        ax.set_xticks(xticks)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax.legend_.remove()
    if fig is not None:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, bbox_to_anchor=(0.5, 1.07), loc="center",
            ncol=2, borderaxespad=0., prop=font_properties)
    ax.grid()


def plot_figure_errors(
    df, x, xlabel, ax, font_properties=None, xscale_log=False, fig=None,
    change_xticks=None, change_yticks=False,
):
    sns.lineplot(
        data=df, x=x, y="Dilations error LBFGSB", linewidth=2.5,
        label="dilations error", estimator=np.median, c=colors[0], linestyle=":",
        marker="^", ax=ax)
    sns.lineplot(
        data=df, x=x, y="Shifts error LBFGSB", linewidth=2.5,
        label="shifts error", estimator=np.median, c=colors[0], linestyle="--",
        marker="o", ax=ax)
    ax.set_xlabel(xlabel, font_properties=font_properties)
    ax.set_ylabel("")
    if xscale_log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    if change_xticks is not None:
        xticks = np.unique(df[x]).astype(int)
        if change_xticks == "half":
            xticks = xticks[2 * np.arange((len(xticks) + 1) // 2)]
        ax.set_xticks(xticks)
    if change_yticks:
        ax.set_yticklabels([], minor=True)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax.legend_.remove()
    if fig is not None:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, bbox_to_anchor=(0.5, 1.02), loc="center",
            fontsize=fontsize, ncol=2, borderaxespad=0., prop=font_properties)
    ax.grid()


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name_1 = f"DataFrame_with_{nb_seeds}_seeds_wrt_n_subjects"
save_name_2 = f"DataFrame_with_{nb_seeds}_seeds_wrt_n_sources"
save_name_3 = f"DataFrame_with_{nb_seeds}_seeds_wrt_n_concat"
save_name_4 = f"DataFrame_with_{nb_seeds}_seeds_wrt_noise"
df_1 = pd.read_csv(results_dir + save_name_1)
df_2 = pd.read_csv(results_dir + save_name_2)
df_3 = pd.read_csv(results_dir + save_name_3)
df_4 = pd.read_csv(results_dir + save_name_4)

# get Times New Roman font
fontsize = 16
font_path = "/storage/store2/work/aheurteb/mvicad/tbme/fonts/Times_New_Roman.ttf"
font_properties = FontProperties(fname=font_path, size=fontsize)

# get colors cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plot Amari distance
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_figure_amari(
    df_1, x="m", xlabel="Number of subjects", ax=axes[0, 0], font_properties=font_properties,
    change_xticks="half")
plot_figure_amari(
    df_2, x="p", xlabel="Number of sources", ax=axes[0, 1], font_properties=font_properties)
plot_figure_amari(
    df_3, x="n_concat", xlabel="Number of concatenations", ax=axes[1, 0],
    font_properties=font_properties, change_xticks="all")
plot_figure_amari(
    df_4, x="noise_data", xlabel="Noise", ax=axes[1, 1], font_properties=font_properties,
    xscale_log=True, fig=fig)
fig.supylabel("Median Amari distance", fontsize=fontsize, font_properties=font_properties)
plt.tight_layout(pad=1.5, w_pad=4.0, h_pad=1.5)

figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_4_figures.pdf", bbox_inches="tight")
plt.show()

# plot shift and dilation's errors
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_figure_errors(
    df_1, x="m", xlabel="Number of subjects", ax=axes[0, 0], font_properties=font_properties,
    change_xticks="half", change_yticks=True)
plot_figure_errors(
    df_2, x="p", xlabel="Number of sources", ax=axes[0, 1], font_properties=font_properties)
plot_figure_errors(
    df_3, x="n_concat", xlabel="Number of concatenations", ax=axes[1, 0], font_properties=font_properties,
    change_xticks="all")
plot_figure_errors(
    df_4, x="noise_data", xlabel="Noise", ax=axes[1, 1], font_properties=font_properties,
    xscale_log=True, fig=fig)
fig.supylabel("Error", fontsize=fontsize, font_properties=font_properties)
plt.tight_layout(pad=1.5, w_pad=4.0, h_pad=1.5)

figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "dilation_shift_errors_4_figures.pdf", bbox_inches="tight")

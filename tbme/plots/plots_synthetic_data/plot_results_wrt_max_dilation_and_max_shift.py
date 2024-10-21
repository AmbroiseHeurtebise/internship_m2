import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
from matplotlib.font_manager import FontProperties
import seaborn as sns


# useful for the legend
class AnyObject(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color


class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        patch = mpl_text.Text(
            x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline',
            horizontalalignment=u'left', multialignment=None,
            fontproperties=None, rotation=0, linespacing=None,
            rotation_mode=None, font_properties=font_properties)
        handlebox.add_artist(patch)
        return patch


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_max_dilation_and_max_shift"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# get Times New Roman font
fontsize = 26
font_path = "/storage/store2/work/aheurteb/mvicad/tbme/fonts/Times_New_Roman.ttf"
font_properties = FontProperties(fname=font_path, size=fontsize)

# colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plot the Amari distance
fig, ax = plt.subplots(figsize=(5, 3.3))
sns.lineplot(
    data=df, x="max_dilation", y="Amari GroupICA", linewidth=2.5,
    label="GroupICA", estimator=np.median, c=colors[4])
sns.lineplot(
    data=df, x="max_dilation", y="Amari MVICA", linewidth=2.5,
    label="MVICA", estimator=np.median, c=colors[2])
sns.lineplot(
    data=df, x="max_dilation", y="Amari permica", linewidth=2.5,
    label="PermICA", estimator=np.median, c=colors[3])
sns.lineplot(
    data=df, x="max_dilation", y="Amari MVICAD ext", linewidth=2.5,
    label="MVICAD", estimator=np.median, c=colors[1])
sns.lineplot(
    data=df, x="max_dilation", y="Amari LBFGSB", linewidth=2.5,
    label="MVICAD$^2$", estimator=np.median, c=colors[0])
# xticks
max_dilation_all = np.unique(df["max_dilation"])
max_shift_all = np.unique(df["max_shift"])
ax.set_xticks(max_dilation_all)
ax.set_xticklabels([])
xticklabels_height = 4 * 1e-6
for i in np.arange(0, len(max_dilation_all), 2):
    ax.text(
        x=max_dilation_all[i], y=xticklabels_height,
        s=f"{max_shift_all[i]:.2f}\n{max_dilation_all[i]:.2f}", ha="center",
        fontsize=fontsize, font_properties=FontProperties(fname=font_path, size=fontsize-2))
ax.text(
    x=0.91, y=xticklabels_height, s="$\\tau_{max}$ :\n$\\rho_{max}$ :", ha="center", fontsize=fontsize,
    font_properties=font_properties)
# yticklabels
for label in ax.get_yticklabels():
    label.set_fontproperties(FontProperties(fname=font_path, size=fontsize-2))
# legend
ax.legend(bbox_to_anchor=(0.98, 1.095), prop=font_properties, handlelength=1.0)
# other features
plt.yscale("log")
ax.set_xlabel("Maximum shift and maximum dilation", font_properties=font_properties)
ax.xaxis.set_label_coords(0.5, -0.36)
ax.set_ylabel("Median Amari\ndistance", font_properties=font_properties)
plt.grid()
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_max_dilation_and_max_shift.pdf", bbox_inches="tight")
plt.show()

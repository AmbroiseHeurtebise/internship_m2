import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
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
            rotation_mode=None)
        handlebox.add_artist(patch)
        return patch


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_synthetic_data/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_wrt_max_dilation_and_max_shift"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plot the Amari distance
plt.figure(figsize=(6, 4))
fig, ax = plt.subplots()
sns.lineplot(
    data=df, x="max_dilation", y="Amari MVICAD", linewidth=2.5,
    label="MVICAD", estimator=np.median, c=colors[1])
sns.lineplot(
    data=df, x="max_dilation", y="Amari MVICAD ext", linewidth=2.5,
    label="MVICAD extended", estimator=np.median, c=colors[2])
sns.lineplot(
    data=df, x="max_dilation", y="Amari permica", linewidth=2.5,
    label="PermICA", estimator=np.median, c=colors[3])
sns.lineplot(
    data=df, x="max_dilation", y="Amari LBFGSB", linewidth=2.5,
    label="LBFGSB", estimator=np.median, c=colors[0])
# xticks
max_dilation_all = np.unique(df["max_dilation"])
max_shift_all = np.unique(df["max_shift"])
xticks = max_dilation_all.copy()
xticklabels = []
for i in range(len(xticks)):
    xticklabels.append(f"{max_dilation_all[i]} | {max_shift_all[i]}")
ax.set_xticks(xticks)
ax.set_xticklabels("")
xticklabels_colors = ["blue", "black", "green"]
for i, tick in enumerate(xticks):
    if i % 2 == 1:
        label_parts = xticklabels[i].split()
        for j, part in enumerate(label_parts):
            ax.text(
                x=tick+0.013*(j-1), y=5.45*1e-5, s=part, transform=ax.transData,
                color=xticklabels_colors[j], fontsize=10, rotation=0, ha="center")
# legend
leg1 = ax.legend(loc="upper left")
ax.add_artist(leg1)
obj_0 = AnyObject("1", xticklabels_colors[0])
obj_1 = AnyObject("2", xticklabels_colors[2])
ax.legend(
    [obj_0, obj_1], ["max_dilation", "max_shift"],
    handler_map={obj_0: AnyObjectHandler(), obj_1: AnyObjectHandler()},
    bbox_to_anchor=(0.99, -0.08))
# other features
plt.yscale("log")
plt.xlabel("Maximum dilation and maximum shift")
ax.xaxis.set_label_coords(0.35, -0.14)
plt.ylabel("Amari distance")
plt.title("Median Amari distance with respect to max_dilation and max_shift")
plt.grid()
plt.subplots_adjust(bottom=0.2)
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_max_dilation_and_max_shift.pdf")
plt.show()

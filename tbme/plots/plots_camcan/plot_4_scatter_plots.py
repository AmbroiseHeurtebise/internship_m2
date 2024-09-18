import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import r2_score


def load_results(task, n_subjects, n_components, n_concat, bis_suffix):
    results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/results/results_camcan" \
        "/mvicad2/clean_subjects/"
    suffix = f"_{task}_task_{n_subjects}_{n_components}_{n_concat}{bis_suffix}.npy"
    W_list = np.load(results_dir + "W" + suffix)
    dilations = np.load(results_dir + "dilations" + suffix)
    shifts = np.load(results_dir + "shifts" + suffix)
    Y_list = np.load(results_dir + "Y" + suffix)
    ages = np.load(results_dir + "ages" + suffix)
    return W_list, dilations, shifts, Y_list, ages


def average_over_sources(dilations_or_shifts, keep_sources):
    return np.mean(dilations_or_shifts[:, keep_sources], axis=1)


def normalize_dilations_and_shifts(dilations, shifts):
    d = 1 / np.mean(dilations)
    s = -np.mean(shifts) * d
    return dilations * d, shifts * d + s


def find_fitLine(dilations_or_shifts, ages):
    slope, intercept, _, _, _ = stats.linregress(ages, dilations_or_shifts)
    fitLine = slope * ages + intercept
    return fitLine, slope, intercept


def find_r2_and_pvalue(dilations, shifts, fitLine_dil, fitLine_shi, ages):
    r2_dil = r2_score(dilations, fitLine_dil)
    r2_shi = r2_score(shifts, fitLine_shi)
    pvalue_dil = stats.pearsonr(ages, dilations)[1]
    pvalue_shi = stats.pearsonr(ages, shifts)[1]
    return r2_dil, r2_shi, pvalue_dil, pvalue_shi


def one_scatter_plot(
    ages, time_params, fitLine, slope, intercept, r2, pvalue, dilations_or_shifts,
    colors, ax=None, sup_ylabel=None, full_title=False, font_properties=None,
):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(ages, time_params)
    ax.plot(ages, fitLine, c=colors[1], linewidth=2)
    if sup_ylabel is not None:
        ax.text(
            x=-0.34, y=0.5, s=sup_ylabel, font_properties=font_properties, rotation=90,
            va="center", transform=ax.transAxes)
    if dilations_or_shifts == "dilations":
        y_hlines = 100
        ylabel = "Dilation (%)"
    elif dilations_or_shifts == "shifts":
        y_hlines = 0
        ylabel = "Shift (ms)"
    else:
        raise ValueError("dilations_or_shifts must be 'dilations' or 'shifts'")
    xmin, xmax = ax.get_xlim()
    ax.hlines(
        y=y_hlines, xmin=xmin, xmax=xmax, linestyles=(5, (10, 3)), colors="black")
    ax.set_xlabel("Age (years)", font_properties=font_properties)
    ax.set_ylabel(ylabel, font_properties=font_properties)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    if full_title:
        if dilations_or_shifts == "dilations":
            sup_title = "Dilation"
        else:
            sup_title = "Shift"
        ax.text(
            x=0.5, y=1.2, s=sup_title, font_properties=font_properties,
            ha="center", transform=ax.transAxes)
    if np.sign(slope) == 1:
        plus_or_minus = "+"
    else:
        plus_or_minus = "-"
    ax.set_title(
        f"R$^2$={r2:.2f} ; m={len(ages)} ; p-value={pvalue:.3f}\n"
        f"y={intercept:.0f}{plus_or_minus}{np.abs(slope):.2f}x",
        font_properties=font_properties)


# parameters
n_subjects = 100
n_components = 8
n_concat = 2
max_dilation_aud = 1.3
max_shift_aud = 0.025
max_dilation_vis = 1.25
max_shift_vis = 0.05
bis_suffix_aud = ""
bis_suffix_vis = "_pen10"
keep_sources_aud = [1, 2, 4]
keep_sources_vis = [2, 5]

# load results
W_list_aud, dilations_aud, shifts_aud, Y_list_aud, ages_aud = load_results(
    "auditory", n_subjects, n_components, n_concat, bis_suffix_aud)
W_list_vis, dilations_vis, shifts_vis, Y_list_vis, ages_vis = load_results(
    "visual", n_subjects, n_components, n_concat, bis_suffix_vis)

# average dilations and shifts
dilations_avg_aud = average_over_sources(dilations_aud, keep_sources_aud)
shifts_avg_aud = average_over_sources(shifts_aud, keep_sources_aud)
dilations_avg_vis = average_over_sources(dilations_vis, keep_sources_vis)
shifts_avg_vis = average_over_sources(shifts_vis, keep_sources_vis)

# change shifts' sign
shifts_avg_aud = -shifts_avg_aud
shifts_avg_vis = -shifts_avg_vis

# normalize dilations and shifts
dilations_avg_aud, shifts_avg_aud = normalize_dilations_and_shifts(
    dilations_avg_aud, shifts_avg_aud)
dilations_avg_vis, shifts_avg_vis = normalize_dilations_and_shifts(
    dilations_avg_vis, shifts_avg_vis)

# change units
factor_dilations = 1e2
factor_shifts = 1e3
dilations_avg_aud *= factor_dilations
shifts_avg_aud *= factor_shifts
dilations_avg_vis *= factor_dilations
shifts_avg_vis *= factor_shifts

# linear regression
fitLine_dil_aud, slope_dil_aud, intercept_dil_aud = find_fitLine(
    dilations_avg_aud, ages_aud)
fitLine_shi_aud, slope_shi_aud, intercept_shi_aud = find_fitLine(
    shifts_avg_aud, ages_aud)
fitLine_dil_vis, slope_dil_vis, intercept_dil_vis = find_fitLine(
    dilations_avg_vis, ages_vis)
fitLine_shi_vis, slope_shi_vis, intercept_shi_vis = find_fitLine(
    shifts_avg_vis, ages_vis)

# r2 score and p-value
r2_dil_aud, r2_shi_aud, pvalue_dil_aud, pvalue_shi_aud = find_r2_and_pvalue(
    dilations_avg_aud, shifts_avg_aud, fitLine_dil_aud, fitLine_shi_aud, ages_aud)
r2_dil_vis, r2_shi_vis, pvalue_dil_vis, pvalue_shi_vis = find_r2_and_pvalue(
    dilations_avg_vis, shifts_avg_vis, fitLine_dil_vis, fitLine_shi_vis, ages_vis)

# color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# get Times New Roman font
fontsize = 20
font_path = "/storage/store2/work/aheurteb/mvicad/tbme/fonts/Times_New_Roman.ttf"
font_properties = FontProperties(fname=font_path, size=fontsize)

# 4 scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
one_scatter_plot(
    ages_aud, shifts_avg_aud, fitLine_shi_aud, slope_shi_aud, intercept_shi_aud,
    r2_shi_aud, pvalue_shi_aud, "shifts", colors, axes[0, 0], sup_ylabel="Auditory",
    full_title=True, font_properties=font_properties)
one_scatter_plot(
    ages_aud, dilations_avg_aud, fitLine_dil_aud, slope_dil_aud, intercept_dil_aud,
    r2_dil_aud, pvalue_dil_aud, "dilations", colors, axes[0, 1], full_title=True,
    font_properties=font_properties)
one_scatter_plot(
    ages_vis, shifts_avg_vis, fitLine_shi_vis, slope_shi_vis, intercept_shi_vis,
    r2_shi_vis, pvalue_shi_vis, "shifts", colors, axes[1, 0], sup_ylabel="Visual",
    font_properties=font_properties)
one_scatter_plot(
    ages_vis, dilations_avg_vis, fitLine_dil_vis, slope_dil_vis, intercept_dil_vis,
    r2_dil_vis, pvalue_dil_vis, "dilations", colors, axes[1, 1],
    font_properties=font_properties)
# plt.tight_layout()
plt.subplots_adjust(wspace=0.38, hspace=0.38)

# save figure
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "scatter_plots_real_data.pdf", bbox_inches="tight")

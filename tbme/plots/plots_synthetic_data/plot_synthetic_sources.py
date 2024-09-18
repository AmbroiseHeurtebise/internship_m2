import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from multiviewica_delay.multiviewica_dilations_shifts import generate_data


# params
m = 5
p = 3
n_concat = 1
n = 600
max_shift = 0.05
max_dilation = 1.15
noise_data = 0.01
n_bins = 10
freq_level = 50
S1_S2_scale = 0.7
onset = 200
random_state = 42
rng = np.random.RandomState(random_state)

# generate synthetic sources
_, _, _, _, _, S = generate_data(
    m=m,
    p=p,
    n_concat=n_concat,
    n=n,
    max_dilation=max_dilation,
    max_shift=max_shift,
    noise_data=noise_data,
    n_bins=n_bins,
    freq_level=freq_level,
    S1_S2_scale=S1_S2_scale,
    rng=rng,
    onset=onset,
)

# get Times New Roman font
fontsize = 18
font_path = "/storage/store2/work/aheurteb/mvicad/tbme/fonts/Times_New_Roman.ttf"
font_properties = FontProperties(fname=font_path, size=fontsize)

# plot
fig, ax = plt.subplots(figsize=(6, 3.5))
height = 0.25
for i in range(p):
    ax.plot(S[i] + height * i)
    ax.hlines(y=height*i, xmin=0, xmax=n, colors="lightgrey")
ax.set_yticks([])
ax.set_yticklabels([])
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)
ax.set_xlabel("Samples", font_properties=font_properties)
ax.set_ylabel("Amplitude", font_properties=font_properties)
plt.grid()
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "synthetic_sources.pdf", bbox_inches="tight")

import numpy as np
# import mne
import os

import matplotlib.pyplot as plt
# from mayavi import mlab
# from tvtk.api import tvtk
# from tvtk.common import configure_input_data
# from surfer import Brain
# from surfer import utils

# os.system("rm ../figures/canica_camcan_montage_*")

# from mne.datasets.sample import data_path

print(__file__)

# rc = {
#     "pdf.fonttype": 42,
#     "text.usetex": True,
#     "font.size": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     # "text.latex.preview": True,
# }
# plt.rcParams.update(rc)


# data_path = data_path()
# subjects_dir = data_path + "/subjects/"
# os.environ["SUBJECTS_DIR"] = subjects_dir

# folder_dir = "drago:/storage/store2/work/hrichard/canica/"
# print("scp " + folder_dir + "stc_avg_496_10_2_audio_task* ../results/canica/")
# os.system("scp " + folder_dir + "stc_avg_496_10_2_audio_task* ../results/canica/")
# print("scp " + folder_dir + "source_reduced_496_10_2_audio_task_0_canica.npy ../results/canica/")
# os.system("scp " + folder_dir + "source_reduced_496_10_2_audio_task_0_canica.npy ../results/canica/")

# S_m = np.load(
#     "../results/canica/source_reduced_496_10_2_audio_task_0_canica.npy"
# )
algo = "mvicad_opt_permica"
save_path = "/storage/store2/work/aheurteb/mvicad/results/%s/" % algo
# save_str = "%d_%d_%d_%s_%s" % (496, 10, 2, "audio", "task",)
save_str = "%d_%d_%d_%s_%s_%i_%s" % (
    100, 10, 2, "audio", "task", 0, algo,)
file_name = save_path + "source_reduced_%s.npy" % save_str
print(file_name)
if os.path.isfile(file_name):
    with open(file_name, "rb") as save_S_file:
        S_m = np.load(save_S_file)
else:
    print("S_m not found\n")
# stc = mne.read_source_estimate("../results/canica/stc_avg_496_10_2_audio_task")

cmap = plt.cm.tab10

print(S_m.shape)
# Plot sources
p, n = S_m.shape

n_times = n // 6
S_m = S_m[:, :n_times]
S_m /= np.sqrt(np.mean(S_m ** 2, axis=1, keepdims=True))
times = np.linspace(-0.2, 0.5, n_times)
# I = np.argsort(np.argmax(np.abs(S_m), axis=1))
# I_inv = np.zeros(len(I))
# I_inv[I] = np.arange(len(I))
# I = I_inv


f, ax = plt.subplots(1, 1, figsize=(6, 2), sharex=True)
for i, s in enumerate(S_m):
    ax.plot(times, s, color=cmap(i)[:3])
# plt.show()
x_ = plt.xlabel("Time (s)")
y_ = plt.ylabel("Amplitude")
plt.xlim([times[0], times[-1]])
plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
plt.savefig(
    "figures/mvicad_opt_permica_camcan_sources_subsample100.pdf",
    bbox_inches="tight",
    bbox_extra_artists=[x_, y_],
)
# plt.show()


# for i in range(p):
#     f, ax = plt.subplots(1, 1, figsize=(3, 2), sharex=True)
#     ax.plot(times, S_m[i], color=cmap(i)[:3])
#     # plt.show()
#     x_ = plt.xlabel("Time (s)")
#     y_ = plt.ylabel("Amplitude")
#     plt.xlim([times[0], times[-1]])
#     plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
#     plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
#     plt.savefig(
#         "../figures/canica_camcan_source_%d.pdf" % I[i],
#         bbox_inches="tight",
#         bbox_extra_artists=[x_, y_],
#     )
#     plt.show()

# for i in range(p):
#     stc_plot = stc.copy()
#     stc_plot = stc_plot.crop(tmin=0.001 * i, tmax=0.001 * i + 0.0005)
#     brain = stc_plot.plot(
#         subject="fsaverage",
#         surface="inflated",
#         subjects_dir=subjects_dir,
#         initial_time=0,
#         hemi="both",
#         views=["lat", "ven", "med"],
#         clim={"kind": "percent", "lims": (90, 95, 100)},
#         time_label=None,
#         colorbar=False,
#     )
#     brain.save_image("../figures/canica_camcan_montage_%i.png" % I[i])

import numpy as np
import os
import matplotlib.pyplot as plt


# Load sources
savefile_name1 = "data/sources1.npy"
savefile_name2 = "data/sources2.npy"
if os.path.isfile(savefile_name1) and os.path.isfile(savefile_name2):
    S1 = np.load(savefile_name1)
    S2 = np.load(savefile_name2)
else:
    print("No sources available \n")

# Correct sign and permutation
n_sources, n_samples = S1.shape
S1 *= (2 * (S1.max(axis=1) > -S1.min(axis=1)) - 1).reshape((n_sources, 1))
S2 *= (2 * (S2.max(axis=1) > -S2.min(axis=1)) - 1).reshape((n_sources, 1))
S2[2] *= -1
S2 = S2[[5, 0, 1, 2, 4, 3], :]

# Plot
time = np.linspace(-0.2, 0.5, n_samples)
plt.figure(figsize=(10, 10))
plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(time, S1[i], label="Early")
    plt.plot(time, S2[i], label="Late")
    plt.legend(prop={'size': 7})
    x_ = plt.xlabel("Time (s.)")
    y_ = plt.ylabel("Amplitude")
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("internship_report_figures/late_early_sources.pdf")

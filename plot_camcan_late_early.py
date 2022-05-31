import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


savefile_name = "results_camcan_late_early.pkl"
if os.path.isfile(savefile_name):
    with open(savefile_name, "rb") as save_ica_file:
        results = pickle.load(save_ica_file)
else:
    print("No sources available \n")

s1 = results["sources1"]
s2 = results["sources2"]

# correct sign

s1 *= 2 * (s1.max(axis=0) > - s1.min(axis=0)) - 1
s2 *= 2 * (s2.max(axis=0) > - s2.min(axis=0)) - 1

source_indexes = [2, 4]

plt.figure(figsize=(3, 2))

n, _ = s1.shape
time = np.linspace(-.2, .5, n)
for sources, index, name in zip([s1, s2], source_indexes, ['Early', 'Late']):
    plt.plot(time, sources[:, index], label=name)
plt.legend()
x_ = plt.xlabel('Time (s.)')
y_ = plt.ylabel('Amplitude')
plt.savefig('camcan_late_early.pdf', bbox_extra_artists=[x_, y_], bbox_inches='tight')


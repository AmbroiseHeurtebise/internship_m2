import pickle

import numpy as np
from mvlearn.decomposition import MultiviewICA


X = np.load('evoked_data_camcan.npy')  ## Camcan data

X /= np.mean(X ** 2, axis=(1, 2), keepdims=True)
power = np.mean(X ** 2, axis=1)
max_idx = np.argmax(power, axis=1)  # peak power index per subject

c1 = (max_idx > 290) * (max_idx < 299)  # Early peak
c2 = (max_idx > 315) * (max_idx < 340)  # Late peak

X1 = X[c1]
X2 = X[c2]
n_sources = 10
ica1 = MultiviewICA(n_sources, n_jobs=1).fit(X1.swapaxes(1, 2))
ica2 = MultiviewICA(n_sources, n_jobs=1).fit(X2.swapaxes(1, 2))

results = {'sources1': ica1.source_, 'sources2': ica2.source_}
with open("results_camcan_late_early.pkl", "wb") as save_results_file:
    pickle.dump(results, save_results_file)

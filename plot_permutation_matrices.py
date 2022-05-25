import numpy as np
import os
import matplotlib.pyplot as plt
from picard import amari_distance
from multiviewica import multiviewica
import pickle


def plot_permutation_matrices(W_approx, A, amari_distances, with_delay):
    n_subjects = len(A)
    fig, ax = plt.subplots(n_subjects // 2, 2)
    fig.suptitle("Permutation matrices " + with_delay + " delay")
    for (i, w, a) in zip(np.arange(n_subjects), W_approx, A):
        plt.subplot(n_subjects // 2, 2, i+1)
        plt.imshow(np.abs(np.dot(w, a)))
        dist = amari_distances[i]
        plt.title("Am. dist. = {:.4f}".format(dist))
        plt.axis('off')
    plt.savefig(with_delay + '.pdf')
    plt.show()
    # idea: plots must represent the same permutation matrix 
    # if there is a delay, these matrices are different, hence the bad reconstruction
    # yet, the Amari distances are low (normally)


if __name__ == '__main__':
    # Load results
    savefile_name = "results_ica"
    if os.path.isfile(savefile_name):
        save_results_file = open(savefile_name, "rb")
        results = pickle.load(save_results_file)
        save_results_file.close()
    else:
        print("There isn't any source to plot \n")

    # Mixing matrices, unmixing matrices and Amari distances for MVICA without delay and with maximum delay
    delay = np.unique(results['Delay'])
    random_states = np.unique(results['random_state'])

    expe_without = results.loc[(results['Algo'] == 'MVICA') & (results['Delay'] == delay[0]) & (results['random_state'] == random_states[0])]
    expe_with = results.loc[(results['Algo'] == 'MVICA') & (results['Delay'] == delay[-1]) & (results['random_state'] == random_states[0])]

    A_without = np.array(expe_without['Mixing'])[0]
    A_with = np.array(expe_with['Mixing'])[0]
    W_without = np.array(expe_without['Unmixing'])[0]
    W_with = np.array(expe_with['Unmixing'])[0]
    amari_without = np.array(expe_without['Amari'])[0]
    amari_with = np.array(expe_with['Amari'])[0]

    # Plots
    plot_permutation_matrices(W_without, A_without, amari_without, with_delay='without')
    plot_permutation_matrices(W_with, A_with, amari_with, with_delay='with')

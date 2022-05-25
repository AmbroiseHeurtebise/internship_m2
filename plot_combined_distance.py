import numpy as np
import matplotlib.pyplot as plt
import os
from picard import amari_distance
import pickle


def combined_distances(W, A):
    mean_amari_distance = np.mean([amari_distance(w, a) for (w, a) in zip(W, A)])
    P = [np.dot(w, a) for (w, a) in zip(W, A)]
    P_normalized = [p / np.outer(np.max(np.abs(p), axis=1), np.ones(np.shape(A)[1])) for p in P]
    dist_Pi = np.mean([np.linalg.norm(Pi - Pj) ** 2 for Pi in P_normalized for Pj in P_normalized])
    return mean_amari_distance + dist_Pi


def plot_distances(delay, distances):
    # Median and std
    # mean_amari = np.mean(distances, axis=2)
    median_amari = np.median(distances, axis=2)
    std_amari = np.std(distances, axis=2)

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Quantity of delay')
    ax.set_ylabel('Amari distance')

    eps = 1e-3
    ax.plot(delay-1, median_amari[:, 0], label='MVICA', c='r')
    low_curve = median_amari[:, 0] - std_amari[:, 0]
    ax.fill_between(delay-1, low_curve * (low_curve > 0) + eps * (low_curve <= 0), median_amari[:, 0] + std_amari[:, 0], color='r', alpha=.1)
    ax.plot(delay-1, median_amari[:, 1], label='GroupICA', color='b')
    low_curve = median_amari[:, 1] - std_amari[:, 1]
    ax.fill_between(delay-1, low_curve * (low_curve > 0) + eps * (low_curve <= 0), median_amari[:, 1] + std_amari[:, 1], color='b', alpha=.1)

    # ax.set_ylim(bottom=0)
    ax.set_yscale('log')
    ax.legend()
    plt.title('Combined distance wrt the quantity of delay', fontweight="bold")
    plt.savefig('combined_distance.pdf')
    plt.show()


if __name__ == '__main__':
    # Load results
    savefile_name = "results_ica"
    if os.path.isfile(savefile_name):
        save_results_file = open(savefile_name, "rb")
        results = pickle.load(save_results_file)
        save_results_file.close()
    else:
        print("There isn't any source to plot \n")

    # Compute combined distance
    results['Combined'] = [combined_distances(w, a) for (w, a) in zip(results['Unmixing'], results['Mixing'])]

    # Get the delays and the distances
    delay = np.unique(results['Delay'])
    algos = ['MVICA', 'GroupICA']
    random_states = np.unique(results['random_state'])

    distances = np.zeros((len(delay), len(algos), len(random_states)))
    for i in np.arange(len(delay)):
        for j in np.arange(len(algos)):
            distances[i, j, :] = results.loc[(results['Delay'] == delay[i]) & (results['Algo'] == algos[j])]['Combined']

    # Plot
    plot_distances(delay, distances)

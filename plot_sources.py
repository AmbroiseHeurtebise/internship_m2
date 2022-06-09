import os
import matplotlib.pyplot as plt
import pickle


def plot_sources(S):
    for i in range(len(S)):
        plt.plot(S[i] + 30 * i)
    plt.title('Sources for the first experiment')
    plt.xlabel('Samples')
    plt.yticks([])
    plt.savefig('figures/sources.pdf')


if __name__ == '__main__':
    # Load results
    savefile_name = "data/results_ica.pkl"
    if os.path.isfile(savefile_name):
        save_results_file = open(savefile_name, "rb")
        results = pickle.load(save_results_file)
        save_results_file.close()
    else:
        print("There isn't any source to plot \n")

    # Get the sources of the first experiment
    S = results.loc[0]['Sources']

    # Plot sources
    plot_sources(S)

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    # Load results
    savefile_name = "data/results_ica.pkl"
    if os.path.isfile(savefile_name):
        with open(savefile_name, "rb") as save_results_file:
            results = pickle.load(save_results_file)
    else:
        print("There isn't any source to plot \n")

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Delay", y="Mean_Amari", hue="Algo", linewidth=2.5)
    fig.set(yscale='log')
    x_ = plt.xlabel("Delay")
    y_ = plt.ylabel("Reconstruction error")
    leg = plt.legend(prop={'size': 20})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Reconstruction error wrt the delay", fontsize=18, fontweight="bold")
    plt.savefig("figures/comparison_algos_ica.pdf", bbox_extra_artists=[x_, y_], bbox_inches="tight")

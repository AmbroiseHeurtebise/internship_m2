import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # Parameters
    m = 10
    p = 5
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 10

    # Load results
    file_name = "m" + str(m) + "_p" + str(p) + "_nbintervals" + str(nb_intervals) + "_nbfreqs" + str(nb_freqs) + "_delaymax" + str(delay_max)
    results = pd.read_csv("data/" + file_name + ".csv")

    # Plot delay error
    plt.figure(1)
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error init", linewidth=2.5, label="Phase 1")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error final", linewidth=2.5, label="Phases 1 and 2")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Error with f", linewidth=2.5, label="Phases 1 and 2 with f")
    fig.set(xscale='log')
    fig.set(yscale='log')
    x_ = plt.xlabel("Noise")
    y_ = plt.ylabel("Error")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Delay estimation error for various noise levels")
    plt.savefig(
        "figures/delay_error_" + file_name + ".pdf", bbox_extra_artists=[x_, y_],
        bbox_inches="tight")

    # Plot Amari distance
    plt.figure(2)
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Noise",
                       y="Amari init", linewidth=2.5, label="Phase 1")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Amari", linewidth=2.5, label="Phases 1 and 2")
    fig = sns.lineplot(data=results, x="Noise",
                       y="Amari with f", linewidth=2.5, label="Phases 1 and 2 with f")
    fig.set(xscale='log')
    fig.set(yscale='log')
    x_ = plt.xlabel("Noise")
    y_ = plt.ylabel("Amari distance")
    leg = plt.legend(prop={'size': 15})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Amari distance for various noise levels")
    plt.savefig(
        "figures/amari_" + file_name + ".pdf", bbox_extra_artists=[x_, y_],
        bbox_inches="tight")

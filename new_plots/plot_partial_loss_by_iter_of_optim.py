import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed
from delay_multiviewica import _optimization_tau, _create_sources, _loss_delay, _optimization_tau_approach1, _optimization_tau_approach2

def create_sources_dummy(n_sub, p, n, delay_max, noise_sources, random_state=None):
    S_list = np.random.randn


def run_experiment(n_sub, p, n, sources_noise, n_iter, random_state, init):
    # Generate data
    _, _, true_tau_list, S_list = _create_sources(
        n_sub, p, n, delay_max=int(0.9*n), noise_sources=sources_noise,
        random_state=random_state)

    loss_true = _loss_delay(S_list, true_tau_list)

    # Optimization
    if init == 'approach_1':
        loss, _, _ = _optimization_tau_approach1(S_list, n_iter)
    elif init == 'approach_2':
        loss, _, _ = _optimization_tau_approach2(S_list, n_iter)
    elif init == 'combined_approach':
        loss, _, _ = _optimization_tau(S_list, n_iter)
    else:
        raise ValueError("init should be approach_1, approach_2 or combined_approach")
    loss = np.array(loss)

    output = {"Init": init, "random_state": random_state,
              "True loss": loss_true, "Loss": loss}
    return output


if __name__ == '__main__':
    # Parameters
    n_sub = 3
    p = 5
    n = 400
    sources_noise = 1
    init = ['approach_1', 'approach_2', 'combined_approach']
    n_iter = 4
    nb_expe = 8
    random_states = 10 + np.arange(nb_expe)
    N_JOBS = 8

    # Results
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(n_sub, p, n, sources_noise, n_iter, r, i)
        for r, i
        in product(random_states, init)
    )
    results = pd.DataFrame(results)
    loss = pd.DataFrame(data=[x for x in results['Loss'].values])
    loss = loss - np.min(loss.min().to_numpy())  # For the sub-optimality
    results = pd.concat([results.drop(columns=['Loss']), loss], axis=1)
    results = results.melt(
        id_vars=["Init", "random_state", "True loss"],
        var_name="Iteration",
        value_name="Loss")

    mean_loss_true = np.mean(results['True loss'])

    # Plot
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style('ticks')
    fig = sns.lineplot(data=results, x="Iteration",
                       y="Loss", hue="Init", linewidth=2.5)
    plt.plot([mean_loss_true] * (n_iter * n_sub + 1),
             '--', color='grey', label='Baseline')
    fig.set(yscale='log')
    x_ = plt.xlabel("Iterations (of the outer loop)")
    y_ = plt.ylabel("Partial loss")
    plt.xticks(np.arange(n_iter + 1) * n_sub, np.arange(n_iter + 1))
    leg = plt.legend(prop={'size': 10})
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    plt.grid()
    plt.title("Partial loss optimization for 3 different approaches",
              fontsize=18, fontweight="bold")
    plt.savefig("new_figures/partial_loss_by_iter_of_optim.pdf",
                bbox_extra_artists=[x_, y_], bbox_inches="tight")
    plt.show()

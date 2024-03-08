# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from apply_dilations_shifts import apply_both_delays_3d_cyclic, apply_dilations_shifts_3d_jax

# %%
if __name__ == "__main__":
    # params
    m = 10
    p = 5
    n = 600
    max_shift = 0.05
    max_dilation = 1.15
    n_concat = 3  # should divide n
    random_state = 122
    rng = np.random.RandomState(random_state)

    # time
    n_it = 50
    time_forloops = []
    time_vectorize = []
    for _ in range(n_it):
        S_list_3d = rng.randn(m, p, n_concat * n)
        S_list_4d = rng.randn(m, p, n_concat, n)
        dilations = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m, p))
        shifts = rng.uniform(low=-max_shift, high=max_shift, size=(m, p))
        # for loops
        start = time()
        S_list_3d_delayed = apply_both_delays_3d_cyclic(
            S_list_3d, dilations, shifts, max_dilation=max_dilation, max_shift=max_shift,
            shift_before_dilation=False, n_concat=n_concat)
        time_forloops.append(time() - start)
        # vectorize
        start = time()
        # S_list_4d_vectorize = apply_dilations_shifts(
        #     S_list_4d, dilations, shifts, max_dilation=max_dilation, max_shift=max_shift, shift_before_dilation=True)
        S_list_3d_vectorize = apply_dilations_shifts_3d_jax(
            S_list_3d, dilations, shifts, max_dilation=max_dilation, max_shift=max_shift,
            n_concat=n_concat)
        time_vectorize.append(time() - start)
# %%
    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([time_forloops, time_vectorize])
    ax.set(ylabel="Time (s)")
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.setp(bp['fliers'], marker='+')
    box_colors = ['royalblue', 'darkred']
    for i in range(2):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax.add_patch(Polygon(box_coords, facecolor=box_colors[i]))
    ax.set_xticklabels(['for loops', 'vectorize'], rotation=45, fontsize=9)
    plt.title("Execution time of the apply_delay function")
    plt.tight_layout()
    plt.show()

# %%

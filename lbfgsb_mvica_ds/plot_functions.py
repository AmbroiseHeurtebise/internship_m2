import numpy as np
import matplotlib.pyplot as plt


def plot_sources_2d(S):
    plt.figure(figsize=(8, 6))
    plt.plot(S.T)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("Original sources")
    plt.grid()
    plt.show()


def plot_sources_3d(S, dilations=None, shifts=None):
    m, p, n = S.shape
    if dilations is None:
        dilations = np.zeros((m, p))
    if shifts is None:
        shifts = np.zeros((m, p))

    plt.subplots(m, p, figsize=(10*p, 2*m))
    for i in range(m):
        for j in range(p):
            plt.subplot(m, p, p * i + j + 1)
            plt.plot(S[i, j], label=f'a={dilations[i, j]:.3f} ; b={shifts[i, j]:.3f}')
            plt.ylim([np.min(S), np.max(S)])
            if j == 0:
                plt.ylabel(f"Subject {i}")
            if i == 1:
                plt.xlabel(f"Source {j}")
            plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f"True sources S ; there are {m} subjects and {p} sources", fontsize=24)
    plt.show()


def scatter_plot_shifts_or_dilations(
    true_params, estim_params, params_error=None, dilations_not_shifts=True, legend=True
):
    # colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # plot
    m, p = true_params.shape
    for i in range(p):
        plt.scatter(true_params[:, i], estim_params[:, i], c=colors[i], label=f"source {i}")
    if dilations_not_shifts:
        params_name = "dilations"
    else:
        params_name = "shifts"
    plt.title(f"{params_name} ; error={params_error:.3}")
    plt.xlabel(f"True {params_name}")
    plt.ylabel(f"Estimated {params_name}")
    if legend:
        plt.legend()
    # diagonal
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [xmin, xmax], c='k', linestyle='--', label="diagonal")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def plot_params_across_iters(params, dilations_not_shifts=True, legend=True):
    n_iters, m = params.shape  # params correspond to one source
    for i in range(m):
        plt.plot(params[:, i], label=f"subject {i}")  # input: -params or 1/params
    plt.grid()
    if legend:
        plt.legend(loc="right")
    if dilations_not_shifts:
        params_name = "dilations"
    else:
        params_name = "shifts"
    plt.xlabel("Iterations")
    plt.ylabel(f"Estimated {params_name}")
    plt.title(f"{params_name} across iterations")


def plot_sources_3_steps(S_true, S_init, S_lbfgsb, source_number=0):
    """ Plot true sources, sources after permica and sources after LBFGSB """

    plt.subplots(1, 3, figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(S_true[:, source_number].T)
    plt.title(f"Source {source_number} from S_list")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(S_init[:, source_number].T)
    plt.title(f"Source {source_number} from permica")
    plt.xlabel("Samples")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(S_lbfgsb[:, source_number].T)
    plt.title(f"Source {source_number} at the end")
    plt.xlabel("Samples")
    plt.grid()

    plt.suptitle(f"Source {source_number} at different steps")
    plt.tight_layout()
    plt.show()


def plot_amari_across_iters(amari_lbfgsb, amari_rand=None, amari_mvicad=None, amari_mvicad_ext=None):
    plt.figure(figsize=(8, 6))
    plt.plot(amari_lbfgsb, label="LBFGSB")
    xmin, xmax = plt.xlim()
    if amari_rand is not None:
        plt.hlines(y=amari_rand, xmin=xmin, xmax=xmax, linestyles='--', colors='k', label="random")
    if amari_mvicad is not None:
        plt.hlines(y=amari_mvicad, xmin=xmin, xmax=xmax, linestyles='--', colors='dimgrey', label="MVICAD")
    if amari_mvicad_ext is not None:
        plt.hlines(
            y=amari_mvicad_ext, xmin=xmin, xmax=xmax, linestyles='--', colors='lightgrey', label="MVICAD extended")
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Amari distance")
    plt.title("Amari distance across iterations")
    plt.xlim([xmin, xmax])
    plt.legend()
    plt.grid()

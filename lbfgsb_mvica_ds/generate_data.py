import numpy as np
from apply_dilations_shifts import apply_dilations_shifts


def generate_sources_1(p, n, rng=None):
    low_mean = n / 3
    high_mean = 8 * n / 10
    means = (np.arange(p) + 0. * rng.uniform(size=p)) * (high_mean - low_mean) / p + low_mean
    rng.shuffle(means)
    variances = rng.randint(n // 25, n // 10, size=p)
    heights = rng.uniform(0.7, 1.4, size=p)

    def f(x, mean, var):
        t = (x - mean) / var
        t[t > 0] /= 2
        s = -t/2 * np.exp(-t ** 2)
        s[s < 0] /= 2
        return s

    S = np.array(
        [height * f(np.arange(n), mean, var)
         for height, mean, var in zip(heights, means, variances)])
    return S


def generate_sources_2(
    freq_list_all,
    ampl_list_all,
    n,
    shifts,
):
    def generate_source(
        freq_list,
        ampl_list,
        n,
        shift,
    ):
        n_bins = len(freq_list)
        n_samples_per_interval = n // n_bins
        s = np.zeros(n)
        for j, (freq, ampl) in enumerate(zip(freq_list, ampl_list)):
            t = np.linspace(0, 1, n_samples_per_interval)
            sine = ampl * np.sin(t * freq)
            sine *= np.exp(- .1 / (t + 1e-7) ** 2)
            sine *= np.exp(- .1 / (1 - t - 1e-7) ** 2)
            s[j * n_samples_per_interval:(j+1) * n_samples_per_interval] = sine
        n_shift = int(shift * n)
        s = np.roll(s, n_shift)
        window = True
        if window:
            s *= np.hamming(n)
            # ts = np.linspace(0, 1, n)
            # s *= np.exp(- .1 / (ts + 1e-7) ** 2)
            # s *= np.exp(- .1 / (1 - ts - 1e-7) ** 2)
        return s
    S = [
        generate_source(freq_list, ampl_list, n, shift)
        for freq_list, ampl_list, shift
        in zip(freq_list_all, ampl_list_all, shifts)]
    S = np.array(S)
    return S


# functions that mixes generation functions 1 and 2
def generate_data(
    m,
    p,
    n,
    max_shift=0.,
    max_dilation=1.,
    noise_data=0.01,
    n_bins=5,  # should divide n
    freq_level=50,
    S1_S2_scale=0.6,
    rng=None,
    n_concat=1,
):
    if n % n_bins != 0:
        print("n_bins should divide n \n")
    S = []
    for _ in range(n_concat):
        # first sources generation function
        S1 = generate_sources_1(p, n, rng)
        # second sources generation function
        freq_list_all = rng.rand(p, n_bins) * freq_level
        ampl_list_all = rng.randn(p, n_bins) * 4
        shifts = rng.rand(p)
        S2 = generate_sources_2(freq_list_all, ampl_list_all, n, shifts)
        # combine both sources generation functions
        S.append(S1_S2_scale * S1 + (1 - S1_S2_scale) * np.max(S1) / np.max(S2) * S2)
    S = np.array(S)                                        # shape (n_concat, p, n)
    S = np.swapaxes(S, axis1=0, axis2=1)                   # shape (p, n_concat, n)
    # other data
    noise_list = noise_data * rng.randn(m, p, n_concat, n)
    S_list = np.array([S + N for N in noise_list])
    S = S.reshape((p, -1))
    A_list = rng.randn(m, p, p)
    dilations = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m, p))
    shifts = rng.uniform(low=-max_shift, high=max_shift, size=(m, p))
    S_list = apply_dilations_shifts(
        S_list, dilations=dilations, shifts=shifts, max_shift=max_shift,
        max_dilation=max_dilation, shift_before_dilation=True)
    S_list = S_list.reshape((m, p, -1))
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, dilations, shifts, S_list, S

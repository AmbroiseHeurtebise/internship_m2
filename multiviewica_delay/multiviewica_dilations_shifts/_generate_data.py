import numpy as np
from scipy.stats import norm
from scipy.signal.windows import tukey
from ._apply_dilations_shifts import apply_dilations_shifts_3d


# ##################################### First data generation function ######################################
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
    S_list = S_list.reshape((m, p, -1))
    A_list = rng.randn(m, p, p)
    dilations = np.ones((m, p))
    shifts = np.zeros((m, p))
    dilations[1:] = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m-1, p))
    shifts[1:] = rng.uniform(low=-max_shift, high=max_shift, size=(m-1, p))
    S_list = apply_dilations_shifts_3d(
        S_list, dilations=dilations, shifts=shifts, max_shift=max_shift,
        max_dilation=max_dilation, shift_before_dilation=True, n_concat=n_concat)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, dilations, shifts, S_list, S


# ##################################### Second data generation function ######################################
def generate_main_pattern(n, rng):
    lower_bound = n // 20
    upper_bound = 5 * n // 6
    n_inner = upper_bound - lower_bound
    n_samples_per_interval = n_inner // 4
    nb_peaks = rng.randint(low=1, high=4)  # either 1, 2 or 3
    peaks = np.sort(rng.choice(np.arange(4), size=nb_peaks, replace=False))
    s = np.zeros(n)
    for peak in peaks:
        t = np.linspace(0, 1, n_samples_per_interval)
        mu = rng.uniform(low=1/3, high=2/3)
        sigma = rng.uniform(low=0.03, high=0.1)
        height = rng.uniform(low=0.8, high=1.2)
        sign = rng.choice([-1, 1])
        s_peak = norm.pdf(t, loc=mu, scale=sigma)
        s_peak *= sign * height / np.max(s_peak)
        s[lower_bound+peak*n_samples_per_interval: lower_bound+(peak+1)*n_samples_per_interval] = s_peak
    kernel_size = n // 20
    kernel = np.ones(kernel_size) / kernel_size
    s = np.convolve(s, kernel, mode="same")
    return s


def generate_frequencies(n, rng, n_intervals=10):
    n_samples_per_interval = n // n_intervals
    n_samples_last_interval = n - (n_intervals - 1) * n_samples_per_interval
    freqs = rng.uniform(low=10, high=50, size=n_intervals)
    heights = rng.uniform(low=0., high=0.6, size=n_intervals)
    s = np.zeros(n)
    for i, (freq, height) in enumerate(zip(freqs, heights)):
        if i < n_intervals:
            t = np.linspace(0, 1, n_samples_per_interval)
        else:
            t = np.linspace(0, 1, n_samples_last_interval)
        sine = height * np.sin(t * freq)
        sine *= np.exp(- .1 / (t + 1e-7) ** 2)
        sine *= np.exp(- .1 / (1 - t - 1e-7) ** 2)
        if i < n_intervals:
            s[i*n_samples_per_interval: (i+1)*n_samples_per_interval] = sine
        else:
            s[-n_samples_last_interval:] = sine
    shift = rng.randint(low=-n_samples_per_interval//2, high=n_samples_per_interval//2)
    s = np.roll(s, shift)
    return s


def generate_source(n, rng, n_intervals=10):
    s1 = generate_main_pattern(n=n, rng=rng)
    s2 = generate_frequencies(n=n, rng=rng, n_intervals=n_intervals)
    s = tukey(n) * (s1 + s2)
    return s


def generate_sources(p, n, rng):
    return np.array([generate_source(n, rng) for _ in range(p)])


def generate_data_multiple_peaks(
    m,
    p,
    n,
    max_shift=0.,
    max_dilation=1.,
    noise_data=0.01,
    rng=None,
    n_concat=1,
):
    # shared sources
    S = np.array([generate_sources(p=p, n=n, rng=rng) for _ in range(n_concat)])  # shape (n_concat, p, n)
    S = np.swapaxes(S, axis1=0, axis2=1)                                          # shape (p, n_concat, n)
    # other data
    noise_list = noise_data * rng.randn(m, p, n_concat, n)
    S_list = np.array([S + N for N in noise_list])
    S = S.reshape((p, -1))
    S_list = S_list.reshape((m, p, -1))
    A_list = rng.randn(m, p, p)
    dilations = np.ones((m, p))
    shifts = np.zeros((m, p))
    dilations[1:] = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m-1, p))
    shifts[1:] = rng.uniform(low=-max_shift, high=max_shift, size=(m-1, p))
    S_list = apply_dilations_shifts_3d(
        S_list, dilations=dilations, shifts=shifts, max_shift=max_shift,
        max_dilation=max_dilation, shift_before_dilation=True, n_concat=n_concat)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, dilations, shifts, S_list, S

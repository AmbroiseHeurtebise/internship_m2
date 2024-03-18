import numpy as np
from scipy.stats import norm
from scipy.signal.windows import tukey
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


# ##################################### Other data generation function (first try) ######################################
# def generate_gaussian(n, rng):
#     t = np.linspace(0, 1, n, endpoint=False)
#     mu = rng.uniform(low=1/3, high=2/3)
#     sigma = rng.uniform(low=0.01, high=0.07)
#     s = norm.pdf(t, loc=mu, scale=sigma)
#     height = rng.uniform(low=0.7, high=1.3)
#     s *= height / np.max(s)
#     return s, mu, sigma


# def generate_main_frequencies(n, mu, sigma, rng):
#     quantiles = norm.ppf([0.025, 0.975], loc=mu, scale=sigma)
#     low = int(quantiles[0] * n)
#     high = int(quantiles[1] * n)
#     t1 = np.linspace(-1, 0, low)
#     t2 = np.linspace(0, 1, n-high)
#     s = np.zeros(n)
#     freq = rng.uniform(low=25, high=35)
#     height1 = rng.uniform(low=0.3, high=0.4)
#     height2 = rng.uniform(low=0.5, high=0.7)
#     s[:low] = height1 * np.sin(freq * t1)
#     s[high:] = height2 * np.sin(freq * t2)
#     kernel_size = n // 20
#     kernel = np.ones(kernel_size) / kernel_size
#     s = np.convolve(s, kernel, mode="same")
#     return s, quantiles


# def window(n):
#     ts = np.linspace(0, 1, n, endpoint=False)
#     s = np.exp(- .1 / (ts + 1e-7) ** 2)
#     s *= np.exp(- .1 / (1 - ts - 1e-7) ** 2)
#     s *= 1 / np.exp(-0.4) ** 2
#     return s


# def generate_frequencies_by_intervals(n, n_intervals, quantiles, rng):
#     low = int(quantiles[0] * n)
#     high = int(quantiles[1] * n)
#     s_sides = np.zeros(n - high + low)
#     n_samples_per_interval = (n - high + low) // n_intervals
#     n_samples_last_interval = (n - high + low) - (n_intervals - 1) * n_samples_per_interval
#     freqs = rng.uniform(low=10, high=50, size=n_intervals)
#     heights = rng.uniform(low=0.2, high=0.7, size=n_intervals)
#     for i, (freq, height) in enumerate(zip(freqs, heights)):
#         if i < n_intervals:
#             t = np.linspace(0, 1, n_samples_per_interval)
#         else:
#             t = np.linspace(0, 1, n_samples_last_interval)
#         sine = height * np.sin(t * freq)
#         sine *= np.exp(- .1 / (t + 1e-7) ** 2)
#         sine *= np.exp(- .1 / (1 - t - 1e-7) ** 2)
#         if i < n_intervals:
#             s_sides[i*n_samples_per_interval: (i+1)*n_samples_per_interval] = sine
#         else:
#             s_sides[-n_samples_last_interval:] = sine
#     s = np.zeros(n)
#     s[:low] = s_sides[:low]
#     s[high:] = s_sides[low:]
#     shift = rng.randint(low=-n_samples_per_interval//2, high=n_samples_per_interval//2)
#     s = np.roll(s, shift)
#     s *= tukey(n)  # Tukey window instead of Hamming window
#     return s


# def generate_source(n, rng, n_intervals=10):
#     s1, mu, sigma = generate_gaussian(n, rng)
#     s2, quantiles = generate_main_frequencies(n, mu, sigma, rng)
#     s3 = generate_frequencies_by_intervals(n, n_intervals, quantiles, rng)
#     s = window(n) * (s1 + s2) + s3
#     return s


# def generate_sources(p, n, rng):
#     return np.array([generate_source(n, rng) for _ in range(p)])


# def generate_new_data(
#     m,
#     p,
#     n,
#     max_shift=0.,
#     max_dilation=1.,
#     noise_data=0.01,
#     rng=None,
#     n_concat=1,
# ):
#     # shared sources
#     S = np.array([generate_sources(p=p, n=n, rng=rng) for _ in range(n_concat)])  # shape (n_concat, p, n)
#     S = np.swapaxes(S, axis1=0, axis2=1)                                          # shape (p, n_concat, n)
#     # other data
#     noise_list = noise_data * rng.randn(m, p, n_concat, n)
#     S_list = np.array([S + N for N in noise_list])
#     S = S.reshape((p, -1))
#     A_list = rng.randn(m, p, p)
#     dilations = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m, p))
#     shifts = rng.uniform(low=-max_shift, high=max_shift, size=(m, p))
#     S_list = apply_dilations_shifts(
#         S_list, dilations=dilations, shifts=shifts, max_shift=max_shift,
#         max_dilation=max_dilation, shift_before_dilation=True)
#     S_list = S_list.reshape((m, p, -1))
#     X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
#     return X_list, A_list, dilations, shifts, S_list, S

# ##################################### Other data generation function (second try) ######################################
# def generate_gaussian(n, rng):
#     t = np.linspace(0, 1, n, endpoint=False)
#     mu = rng.uniform(low=1/3, high=2/3)
#     sigma = rng.uniform(low=0.01, high=0.07)
#     s = norm.pdf(t, loc=mu, scale=sigma)
#     height = rng.uniform(low=0.7, high=1.3)
#     s *= height / np.max(s)
#     return s, mu, sigma


# def generate_main_frequencies(n, mu, sigma, rng):
#     quantiles = norm.ppf([0.025, 0.975], loc=mu, scale=sigma)
#     low = int(quantiles[0] * n)
#     high = int(quantiles[1] * n)
#     t1 = np.linspace(-1, 0, low)
#     t2 = np.linspace(0, 1, n-high)
#     s = np.zeros(n)
#     freq = rng.uniform(low=25, high=35)
#     height1 = rng.uniform(low=0.3, high=0.4)
#     height2 = rng.uniform(low=0.5, high=0.7)
#     s[:low] = height1 * np.sin(freq * t1)
#     s[high:] = height2 * np.sin(freq * t2)
#     kernel_size = n // 20
#     kernel = np.ones(kernel_size) / kernel_size
#     s = np.convolve(s, kernel, mode="same")
#     return s, quantiles


# def window(n):
#     ts = np.linspace(0, 1, n, endpoint=False)
#     s = np.exp(- .1 / (ts + 1e-7) ** 2)
#     s *= np.exp(- .1 / (1 - ts - 1e-7) ** 2)
#     s *= 1 / np.exp(-0.4) ** 2
#     return s


# def generate_frequencies_by_intervals(n, n_intervals, quantiles, rng):
#     low = int(quantiles[0] * n)
#     high = int(quantiles[1] * n)
#     s_sides = np.zeros(n - high + low)
#     n_samples_per_interval = (n - high + low) // n_intervals
#     n_samples_last_interval = (n - high + low) - (n_intervals - 1) * n_samples_per_interval
#     freqs = rng.uniform(low=10, high=50, size=n_intervals)
#     heights = rng.uniform(low=0.2, high=0.7, size=n_intervals)
#     for i, (freq, height) in enumerate(zip(freqs, heights)):
#         if i < n_intervals:
#             t = np.linspace(0, 1, n_samples_per_interval)
#         else:
#             t = np.linspace(0, 1, n_samples_last_interval)
#         sine = height * np.sin(t * freq)
#         sine *= np.exp(- .1 / (t + 1e-7) ** 2)
#         sine *= np.exp(- .1 / (1 - t - 1e-7) ** 2)
#         if i < n_intervals:
#             s_sides[i*n_samples_per_interval: (i+1)*n_samples_per_interval] = sine
#         else:
#             s_sides[-n_samples_last_interval:] = sine
#     s = np.zeros(n)
#     s[:low] = s_sides[:low]
#     s[high:] = s_sides[low:]
#     shift = rng.randint(low=-n_samples_per_interval//2, high=n_samples_per_interval//2)
#     s = np.roll(s, shift)
#     s *= tukey(n)  # Tukey window instead of Hamming window
#     return s


# def generate_source(n, rng, n_intervals=10):
#     s1, mu, sigma = generate_gaussian(n, rng)
#     s2, quantiles = generate_main_frequencies(n, mu, sigma, rng)
#     s3 = generate_frequencies_by_intervals(n, n_intervals, quantiles, rng)
#     s = window(n) * (s1 + s2) + s3
#     return s


# def generate_sources(p, n, rng):
#     return np.array([generate_source(n, rng) for _ in range(p)])


# def generate_new_data(
#     m,
#     p,
#     n,
#     max_shift=0.,
#     max_dilation=1.,
#     noise_data=0.01,
#     rng=None,
#     n_concat=1,
# ):
#     # shared sources
#     S = np.array([generate_sources(p=p, n=n, rng=rng) for _ in range(n_concat)])  # shape (n_concat, p, n)
#     S = np.swapaxes(S, axis1=0, axis2=1)                                          # shape (p, n_concat, n)
#     # other data
#     noise_list = noise_data * rng.randn(m, p, n_concat, n)
#     S_list = np.array([S + N for N in noise_list])
#     S = S.reshape((p, -1))
#     A_list = rng.randn(m, p, p)
#     dilations = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m, p))
#     shifts = rng.uniform(low=-max_shift, high=max_shift, size=(m, p))
#     S_list = apply_dilations_shifts(
#         S_list, dilations=dilations, shifts=shifts, max_shift=max_shift,
#         max_dilation=max_dilation, shift_before_dilation=True)
#     S_list = S_list.reshape((m, p, -1))
#     X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
#     return X_list, A_list, dilations, shifts, S_list, S


# ##################################### Other data generation function (last try) ######################################
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


def generate_new_data(
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
    A_list = rng.randn(m, p, p)
    dilations = rng.uniform(low=1/max_dilation, high=max_dilation, size=(m, p))
    shifts = rng.uniform(low=-max_shift, high=max_shift, size=(m, p))
    S_list = apply_dilations_shifts(
        S_list, dilations=dilations, shifts=shifts, max_shift=max_shift,
        max_dilation=max_dilation, shift_before_dilation=True)
    S_list = S_list.reshape((m, p, -1))
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, dilations, shifts, S_list, S

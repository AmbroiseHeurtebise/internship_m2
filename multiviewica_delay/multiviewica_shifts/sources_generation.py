import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.utils import check_random_state
from multiviewica_delay import (
    _apply_delay,
    _apply_delay_one_sub,
    _apply_delay_by_source,
    _apply_delay_one_source_or_sub,
)


# ------------- Old data generation functions -------------
def _create_sources(n_sub, p, n, max_delay=None, noise_sources=0.05, random_state=None):
    rng = check_random_state(random_state)
    if max_delay is None:
        max_delay = int(0.2 * n)
    n_inner = n - max_delay
    window = np.hamming(n_inner)
    S = window * np.outer(np.ones(p),
                          np.sin(np.linspace(0, 4 * np.pi, n_inner)))
    flip = np.outer(rng.randint(0, 2, p), np.ones(n_inner)) * 2 - 1
    S *= flip
    noise_list = noise_sources * rng.randn(n_sub, p, n_inner)
    S_list = np.array([S + n for n in noise_list])
    S_list = np.concatenate(
        (S_list, np.zeros((n_sub, p, n - n_inner))), axis=2)
    tau_list = rng.randint(0, max_delay, size=n_sub)
    S_list = _apply_delay(S_list, tau_list)
    A_list = rng.randn(n_sub, p, p)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list


def create_sources_pierre(m, p, n, max_delay, sigma=0.05, random_state=None):
    rng = check_random_state(random_state)
    delays = rng.randint(0, 1 + max_delay, m)
    s1 = np.zeros(n)
    s1[:n//2] = np.sin(np.linspace(0, np.pi, n//2))
    s2 = rng.randn(n) / 10
    S = np.c_[s1, s2].T
    N = sigma * rng.randn(m, p, n)
    A_list = rng.randn(m, p, p)
    X_list = np.array([np.dot(A, _apply_delay_one_sub(S, -delay) + noise)
                      for A, noise, delay in zip(A_list, N, delays)])
    return X_list, A_list, delays, S
# Only works for p=2


def create_model(m, p, n, delay=None, noise=0.05, random_state=None):
    rng = check_random_state(random_state)
    if delay is None:
        delay = int(0.2 * n)
    s = np.zeros(n)
    s[:n//4] = np.sin(np.linspace(0, np.pi, n//4))
    delay_sources = rng.randint(0, n, p)
    S = np.array([np.roll(s, -delay_sources[i]) for i in range(p)])
    noise_list = noise * rng.randn(m, p, n)
    S_list = np.array([S + N for N in noise_list])
    tau_list = rng.randint(0, delay + 1, size=m)
    S_list = _apply_delay(S_list, tau_list)
    A_list = rng.randn(m, p, p)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list, S


# ------------- Actual data generation functions -------------
def soft_treshold(S, treshold=1):
    return np.sign(S) * np.maximum(0, np.abs(S) - treshold)


def generate_source_one_interval(interval_length, freqs):
    t = np.linspace(0, 6 * np.pi, interval_length)
    s = np.sum([np.sin(f * t) for f in freqs], axis=0)
    return s


def generate_one_source(interval_length, freqs, power):
    s = np.array([p * generate_source_one_interval(interval_length, freqs[i, :]) for p, i in zip(power, range(len(power)))]).reshape(-1)
    return s


def generate_sources(p, n, nb_intervals=5, nb_freqs=20, random_state=None):
    rng = check_random_state(random_state)
    interval_length = n // nb_intervals
    freqs = rng.randn(p, nb_intervals, nb_freqs)
    power = rng.exponential(size=(p, nb_intervals))
    S = np.array([generate_one_source(interval_length, freqs[i], power[i]) for i in range(p)])
    shifts = (rng.rand(p) * n).astype("int")
    S = _apply_delay_one_source_or_sub(S, shifts)
    return S


def generate_data(
    m,
    p,
    n,
    nb_intervals=5,
    nb_freqs=20,
    treshold=1,
    delay=None,
    noise=0.05,
    random_state=None,
    shared_delays=True,
):
    rng = check_random_state(random_state)
    if delay is None:
        delay = n // 5
    A_list = rng.randn(m, p, p)
    S = generate_sources(p, n, nb_intervals, nb_freqs, random_state)
    S = soft_treshold(S, treshold=treshold)
    noise_list = noise * rng.randn(m, p, n)
    S_list = np.array([S + N for N in noise_list])
    if shared_delays:
        # tau_list = rng.randint(0, delay + 1, size=m)
        # XXX should sample between -max_delay and max_delay, not between 0 and max_delay
        tau_list = rng.randint(-delay, delay + 1, size=m)
        tau_list[0] = 0   # XXX
        S_list = _apply_delay(S_list, tau_list)
    else:
        # tau_list = rng.randint(0, delay + 1, size=(m, p))
        # XXX should sample between -max_delay and max_delay, not between 0 and max_delay
        tau_list = rng.randint(-delay, delay + 1, size=(m, p))
        tau_list[0] = 0   # XXX
        S_list = _apply_delay_by_source(S_list, tau_list)
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list, S


# ------------- New data generation functions -------------
def sources_generation(p, n, random_state=None):
    rng = check_random_state(random_state)
    means = rng.randint(n // 4, 3 * n // 4, size=p)
    variances = rng.randint(n // 25, n // 10, size=p)
    S = np.array(
        [norm.pdf(np.arange(n), mean, var)
         for mean, var in zip(means, variances)])
    return S


def sources_generation_zero_mean(p, n, random_state=None):
    rng = check_random_state(random_state)
    # means = rng.randint(n // 4, 3 * n // 4, size=p)
    means = np.cumsum(rng.poisson(lam=n/(2*p), size=p)) + n // 6
    rng.shuffle(means)
    variances = rng.randint(n // 25, n // 8, size=p)
    heights = rng.randint(80, 120, size=p) / 100

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


def delays_generation(m, max_delay, random_state=None):
    rng = check_random_state(random_state)
    delays = (1/2 * max_delay * rng.randn(m)).astype('int')
    nb_outliers = np.sum(np.abs(delays) > max_delay)
    while(nb_outliers > 0):
        delays[np.abs(delays) > max_delay] = (
            1/2 * max_delay * rng.randn(nb_outliers)).astype('int')
        nb_outliers = np.sum(np.abs(delays) > max_delay)
    return delays


def data_generation(
    m,
    p,
    n,
    max_delay=0,
    noise=5*1e-4,
    shared_delays=True,
    random_state=None,
    n_concat=1,
):
    rng = check_random_state(random_state)
    # S = sources_generation(p, n, rng)
    S = sources_generation_zero_mean(p, n, rng)
    S = np.hstack([S] * n_concat)
    noise_list = noise * rng.randn(m, p, n * n_concat)
    S_list = np.array([S + N for N in noise_list])
    A_list = rng.randn(m, p, p)
    if shared_delays:
        # tau_list = rng.randint(-max_delay, max_delay + 1, size=m)
        tau_list = delays_generation(m, max_delay, rng)
        tau_list[0] = 0   # XXX
        S_list = _apply_delay(S_list, tau_list)
    else:
        # tau_list = rng.randint(-max_delay, max_delay + 1, size=(m, p))
        tau_list = np.array(
            [delays_generation(m, max_delay, rng) for _ in range(p)]).T
        tau_list[0] = 0   # XXX
        S_list = _apply_delay_by_source(S_list, tau_list)
    tau_list %= n
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list, S


# ------------- New data generation functions by Pierre -------------
def data_generation_pierre(
    n_subjects,
    n_sources,
    n_bins,
    n_samples_per_interval,
    freq_level=50,
    max_delay=0,
    noise=0.2,
    shared_delays=False,
    random_state=None,
):
    rng = check_random_state(random_state)

    # frequences, amplitudes and shifts
    freq_list_all = rng.rand(n_sources, n_bins) * freq_level
    ampl_list_all = rng.randn(n_sources, n_bins) * 4
    shifts = rng.rand(n_sources)

    # function written by Pierre
    def generate_sources(
        freq_list_all,
        ampl_list_all,
        n_samples_per_interval,
        shifts,
    ):
        def generate_source(
            freq_list,
            ampl_list,
            n_samples_per_interval,
            shift,
        ):
            n_bins = len(freq_list)
            s = np.zeros(n_bins * n_samples_per_interval)
            for j, (freq, ampl) in enumerate(zip(freq_list, ampl_list)):
                t = np.linspace(0, 1, n_samples_per_interval)
                sine = ampl * np.sin(t * freq)
                sine *= np.exp(- .1 / (t + 1e-7) ** 2)
                sine *= np.exp(- .1 / (1 - t - 1e-7) ** 2)
                s[j * n_samples_per_interval:(j+1) * n_samples_per_interval] = sine
            n_shift = int(shift * n_bins * n_samples_per_interval)
            s = np.roll(s, n_shift)
            window = True
            if window:
                s *= np.hamming(n_bins * n_samples_per_interval)
                # ts = np.linspace(0, 1, n_bins * n_samples_per_interval)
                # s *= np.exp(- .1 / (ts + 1e-7) ** 2)
                # s *= np.exp(- .1 / (1 - ts - 1e-7) ** 2)
            return s
        S = [
            generate_source(freq_list, ampl_list, n_samples_per_interval, shift)
            for freq_list, ampl_list, shift
            in zip(freq_list_all, ampl_list_all, shifts)]
        S = np.array(S)
        return S
    S = generate_sources(
        freq_list_all, ampl_list_all, n_samples_per_interval, shifts)

    # other data
    noise_list = noise * rng.randn(
        n_subjects, n_sources, n_bins * n_samples_per_interval)
    S_list = np.array([S + N for N in noise_list])
    A_list = rng.randn(n_subjects, n_sources, n_sources)
    if shared_delays:
        tau_list = delays_generation(n_subjects, max_delay, rng)
        tau_list[0] = 0   # XXX
        S_list = _apply_delay(S_list, tau_list)
    else:
        tau_list = np.array(
            [delays_generation(n_subjects, max_delay, rng)
             for _ in range(n_sources)]).T
        tau_list[0] = 0   # XXX
        S_list = _apply_delay_by_source(S_list, tau_list)
    tau_list %= n_bins * n_samples_per_interval
    X_list = np.array([np.dot(A, S) for A, S in zip(A_list, S_list)])
    return X_list, A_list, tau_list, S_list, S


# ------------- Functions that plot sources -------------
def plot_sources(S, nb_intervals=None):
    p, n = S.shape
    height = 20
    for i in range(p):
        plt.plot(S[i] + height * i)
        plt.hlines(y=height*i, xmin=0, xmax=n, linestyles='--', colors='grey')
        plt.yticks([])
    if nb_intervals is not None:
        for i in range(nb_intervals+1):
            plt.vlines(x=n//nb_intervals*i, ymin=-height, ymax=p*height,
            linestyles='--', colors='grey')
    plt.xlabel("Samples")
    # plt.title("Sources", fontsize=18, fontweight="bold")
    plt.savefig("internship_report_figures/sources.pdf")


def _plot_delayed_sources(S_list, height=20, nb_intervals=None):
    n_sub, p, n = S_list.shape
    fig, _ = plt.subplots(n_sub//2, 2)
    # fig.suptitle("Delayed sources of {} subjects".format(n_sub), fontsize=18, fontweight="bold")
    for i in range(n_sub):
        plt.subplot(n_sub//2, 2, i+1)
        for j in range(p):
            plt.plot(S_list[i, j, :] + height * j)
            plt.hlines(y=height*j, xmin=0, xmax=n, linestyles='--', colors='grey')
        # plt.grid()
        if nb_intervals is not None:
            for k in range(nb_intervals+1):
                plt.vlines(x=n//nb_intervals*k, ymin=-height, ymax=p*height,
                linestyles='--', colors='grey')
        plt.yticks([])
        plt.title("Sources of subject {}".format(i+1))
        if i >= n_sub - 2:
            plt.xlabel("Samples")
        else:
            plt.xticks([])

    # plt.savefig("figures/sources_wrong_sign_and_order.pdf")
    plt.savefig("internship_report_figures/sources_multiple_subjects.pdf")

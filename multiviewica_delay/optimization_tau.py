import numpy as np


# ------------- Functions that apply delay to signals -------------
def _apply_delay_one_sub(Y, tau):
    return np.roll(Y, tau, axis=1)


def _apply_delay(Y_list, tau_list):
    return np.array(
        [_apply_delay_one_sub(Y, tau)
         for Y, tau in zip(Y_list, tau_list)])


def _apply_delay_one_source_or_sub(S, delays):
    return np.array([np.roll(S[i], delays[i]) for i in range(len(S))])


def _apply_delay_by_source(S_list, delays_by_source):
    Y_list = np.array([
        _apply_delay_one_source_or_sub(
            S_list[i], delays_by_source[i].astype('int'))
        for i in range(len(S_list))])
    return Y_list


# ------------- Loss function -------------
def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))


def _loss_mvica(Y_list, noise=1, use_f=True):
    _, p, _ = Y_list.shape
    Y_avg = np.mean(Y_list, axis=0)
    loss = 0
    if use_f:
        loss += np.mean(_logcosh(Y_avg)) * p
    for Y in Y_list:
        loss += 1 / (2 * noise) * np.mean((Y - Y_avg) ** 2) * p
    return loss


# ------------- Function that compute distance between delays -------------
# could be useful in _multiviewica_delay
# def distance_between_delays(tau1, tau2, n):
#     assert len(tau1) == len(tau2)
#     n_sub = len(tau1)
#     error_total = []
#     for i in range(n):
#         error = 0
#         tau3 = (tau2 + i) % n
#         for j in range(n_sub):
#             error += np.min(np.abs([tau1[j] - tau3[j] - n, tau1[j] - tau3[j], tau1[j] - tau3[j] + n]))
#         error_total.append(error)
#     return np.min(np.array(error_total))


# ------------- Functions used in _optimization_tau methods -------------
def _delay_estimation(Y, Y_avg, tau_i=0, max_delay=10):
    _, n = Y.shape
    if max_delay is not None:
        if tau_i > n // 2:
            tau_i -= n
        conv1 = np.array([np.convolve(
            np.concatenate((y, y[:max_delay-tau_i])), y_avg[::-1], mode='valid')
            for y, y_avg in zip(Y, Y_avg)])
        conv1_norm = np.sum(conv1, axis=0)
        if tau_i != -max_delay:
            conv2 = np.array([np.convolve(
                np.concatenate((y[n-max_delay-tau_i:], y[:-1])), y_avg[::-1], mode='valid')
                for y, y_avg in zip(Y, Y_avg)])
            conv2_norm = np.sum(conv2, axis=0)
        else:
            conv2_norm = np.array([])
        conv_norm = np.concatenate((conv1_norm, conv2_norm))
    else:
        conv = np.array([np.convolve(
            np.concatenate((y, y[:-1])), y_avg[::-1], mode='valid')
            for y, y_avg in zip(Y, Y_avg)])
        conv_norm = np.sum(conv, axis=0)
    optimal_delay = np.argmax(conv_norm)
    if max_delay is not None and optimal_delay > max_delay - tau_i:
        optimal_delay += n - 2 * max_delay - 1
    return optimal_delay


def _delay_estimation_with_f(Y_list, index, tau_i=0, noise=1, use_f=True, max_delay=10):
    _, _, n = Y_list.shape
    Y = Y_list[index].copy()
    Y_list_copy = Y_list.copy()
    loss = []
    if max_delay is not None:
        if tau_i > n // 2:
            tau_i -= n
        delays = np.concatenate((np.arange(max_delay-tau_i+1), np.arange(n-max_delay-tau_i, n)))
    else:
        delays = np.arange(n)
    for delay in delays:
        Y_delayed = _apply_delay_one_sub(Y, -delay)
        Y_list_copy[index] = Y_delayed
        loss.append(_loss_mvica(Y_list_copy, noise, use_f))
    optimal_delay = np.argmin(loss)
    if max_delay is not None and optimal_delay > max_delay - tau_i:
        optimal_delay += n - 2 * max_delay - 1
    return optimal_delay


# ------------- _optimization_tau methods -------------
def _optimization_tau_approach1(
    S_list,
    n_iter,
    max_delay=10,
    tau_list_init=None,
    previous_tau_list=None
):
    n_sub, _, n = S_list.shape
    if previous_tau_list is None:
        previous_tau_list = np.zeros(n_sub, dtype=int)
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_list = _apply_delay(S_list, -previous_tau_list)
    Y_list_freeze = Y_list.copy()
    Y_avg = np.mean(Y_list, axis=0)
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + previous_tau_list + tau_list
    for _ in range(n_iter):
        for i in range(n_sub):
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(
                Y_list[i],
                new_Y_avg,
                tau_i=tau_list_final[i],
                max_delay=max_delay,
            )
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay_one_sub(Y_list_freeze[i], -tau_list[i])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
        tau_list_final = tau_list_init + previous_tau_list + tau_list
        tau_list_final %= n
    return (previous_tau_list + tau_list) % n


def _optimization_tau_approach2(
    S_list,
    n_iter,
    max_delay=10,
    tau_list_init=None,
    previous_tau_list=None
):
    n_sub, _, n = S_list.shape
    if previous_tau_list is None:
        previous_tau_list = np.zeros(n_sub, dtype=int)
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_list = _apply_delay(S_list, -previous_tau_list)
    Y_list_freeze = Y_list.copy()
    Y_avg = Y_list[0]
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + previous_tau_list + tau_list
    for _ in range(n_iter):
        for i in range(n_sub):
            tau_list[i] += _delay_estimation(
                Y_list[i],
                Y_avg,
                tau_i=tau_list_final[i],
                max_delay=max_delay,
            )
            Y_list[i] = _apply_delay_one_sub(Y_list_freeze[i], -tau_list[i])
        Y_avg = np.mean(Y_list, axis=0)
        tau_list %= n
        tau_list_final = tau_list_init + previous_tau_list + tau_list
        tau_list_final %= n
    return (previous_tau_list + tau_list) % n


def _optimization_tau(
    S_list,
    n_iter,
    max_delay=10,
    tau_list_init=None,
    previous_tau_list=None
):
    n_sub, _, n = S_list.shape
    if previous_tau_list is None:
        previous_tau_list = np.zeros(n_sub, dtype=int)
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_list = _apply_delay(S_list, -previous_tau_list)
    Y_list_freeze = Y_list.copy()
    Y_avg = Y_list[0]
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + previous_tau_list + tau_list
    for i in range(n_sub):
        tau_list[i] = _delay_estimation(
            Y_list[i],
            Y_avg,
            tau_i=tau_list_final[i],
            max_delay=max_delay,
        )
        Y_list[i] = _apply_delay_one_sub(Y_list_freeze[i], -tau_list[i])
    Y_avg = np.mean(Y_list, axis=0)
    tau_list %= n
    tau_list_final = tau_list_init + previous_tau_list + tau_list
    tau_list_final %= n
    for _ in range(n_iter-1):
        for i in range(n_sub):
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(
                Y_list[i],
                new_Y_avg,
                tau_i=tau_list_final[i],
                max_delay=max_delay,
            )
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay_one_sub(Y_list_freeze[i], -tau_list[i])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
        tau_list_final = tau_list_init + previous_tau_list + tau_list
        tau_list_final %= n
    return (previous_tau_list + tau_list) % n


def _optimization_tau_with_f(
    S_list,
    n_iter,
    noise=1,
    use_f=True,
    max_delay=10,
    tau_list_init=None,
    previous_tau_list=None
):
    n_sub, _, n = S_list.shape
    if previous_tau_list is None:
        previous_tau_list = np.zeros(n_sub, dtype=int)
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_list = _apply_delay(S_list, -previous_tau_list)
    Y_list_freeze = Y_list.copy()
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + previous_tau_list + tau_list
    for _ in range(n_iter):
        for i in range(n_sub):
            tau_list[i] += _delay_estimation_with_f(
                Y_list,
                i,
                tau_i=tau_list_final[i],
                noise=noise,
                use_f=use_f,
                max_delay=max_delay,
            )
            tau_list[i] %= n
            Y_list[i] = _apply_delay_one_sub(Y_list_freeze[i], -tau_list[i])
        tau_list_final = tau_list_init + previous_tau_list + tau_list
        tau_list_final %= n
    return (previous_tau_list + tau_list) % n


def _optimization_tau_by_source(
    S_list,
    n_iter=3,
    max_delay=20,
    previous_tau_list=None,
    use_loss_total=False,
    noise=1,
):
    _, p, _ = S_list.shape
    tau_list = previous_tau_list.copy()
    for i in range(p):
        sources = S_list[:, i]
        sources = sources[:, None, :]
        if use_loss_total:
            estimated_delays = _optimization_tau_with_f(
                sources,
                n_iter=n_iter,
                noise=noise,
                max_delay=max_delay,
                previous_tau_list=previous_tau_list[:, i])
        else:
            estimated_delays = _optimization_tau_approach1(
                sources,
                n_iter=n_iter,
                max_delay=max_delay,
                previous_tau_list=previous_tau_list[:, i])
        tau_list[:, i] = estimated_delays
    return tau_list

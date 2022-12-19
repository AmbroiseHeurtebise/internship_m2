import numpy as np


def _apply_delay_one_sub(Y, tau):
    return np.roll(Y, tau, axis=1)


def _apply_delay(Y_list, tau_list):
    return np.array([_apply_delay_one_sub(Y, tau) for Y, tau in zip(Y_list, tau_list)])


def _loss_function(Y_list, Y_avg):
    loss = np.mean((Y_list - Y_avg) ** 2)
    return loss


def _loss_delay(S_list, tau_list):
    Y_list = _apply_delay(S_list, -tau_list)
    Y_avg = np.mean(Y_list, axis=0)
    return _loss_function(Y_list, Y_avg)


def _loss_delay_ref(S_list, tau_list, Y_avg):
    Y_list = _apply_delay(S_list, -tau_list)
    return _loss_function(Y_list, Y_avg)


def _delay_estimation(Y, Y_avg, tau_i=0, delay_max=10):
    _, n = Y.shape
    if delay_max is not None:
        if tau_i > n // 2:
            tau_i -= n
        conv1 = np.array([np.convolve(
            np.concatenate((y, y[:delay_max-tau_i])), y_avg[::-1], mode='valid')
            for y, y_avg in zip(Y, Y_avg)])
        conv1_norm = np.sum(conv1, axis=0)
        if tau_i != -delay_max:
            conv2 = np.array([np.convolve(
                np.concatenate((y[n-delay_max-tau_i:], y[:-1])), y_avg[::-1], mode='valid')
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
    if delay_max is not None and optimal_delay > delay_max - tau_i:
        optimal_delay += n - 2 * delay_max - 1
    return optimal_delay


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


def _delay_estimation_with_f(Y_list, index, tau_i=0, noise=1, use_f=True, delay_max=10):
    _, _, n = Y_list.shape
    Y = Y_list[index].copy()
    Y_list_copy = Y_list.copy()
    loss = []
    if delay_max is not None:
        if tau_i > n // 2:
            tau_i -= n
        delays = np.concatenate((np.arange(delay_max-tau_i+1), np.arange(n-delay_max-tau_i, n)))
    else:
        delays = np.arange(n)
    for delay in delays:
        Y_delayed = _apply_delay_one_sub(Y, -delay)
        Y_list_copy[index] = Y_delayed
        loss.append(_loss_mvica(Y_list_copy, noise, use_f))
    optimal_delay = np.argmin(loss)
    if delay_max is not None and optimal_delay > delay_max - tau_i:
        optimal_delay += n - 2 * delay_max - 1
    return optimal_delay


def distance_between_delays(tau1, tau2, n):
    assert len(tau1) == len(tau2)
    n_sub = len(tau1)
    error_total = []
    for i in range(n):
        error = 0
        tau3 = (tau2 + i) % n
        for j in range(n_sub):
            error += np.min(np.abs([tau1[j] - tau3[j] - n, tau1[j] - tau3[j], tau1[j] - tau3[j] + n]))
        error_total.append(error)
    return np.min(np.array(error_total))


def _optimization_tau_approach1(
    S_list, n_iter, delay_max=10, error_tau=False, true_tau_list=None,
    tau_list_init=None
):
    n_sub, _, n = S_list.shape
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_avg = np.mean(S_list, axis=0)
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + tau_list
    loss = []
    gap_tau = []
    for _ in range(n_iter):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau is True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(
                Y_list[i], new_Y_avg, tau_i=tau_list_final[i], delay_max=delay_max)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([S_list[i]], [-tau_list[i]])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
        tau_list_final = tau_list_init + tau_list
        tau_list_final %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau is True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau_approach2(
    S_list, n_iter, delay_max=10, error_tau=False, true_tau_list=None,
    tau_list_init=None
):
    n_sub, _, n = S_list.shape
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_avg = S_list[0]
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + tau_list
    loss = []
    gap_tau = []
    for _ in range(n_iter):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau is True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            tau_list[i] += _delay_estimation(
                Y_list[i], Y_avg, tau_i=tau_list_final[i], delay_max=delay_max)
            Y_list[i] = _apply_delay([S_list[i]], [-tau_list[i]])
        Y_avg = np.mean(Y_list, axis=0)
        tau_list %= n
        tau_list_final = tau_list_init + tau_list
        tau_list_final %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau is True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau(
    S_list, n_iter, delay_max=10, error_tau=False, true_tau_list=None,
    tau_list_init=None
):
    n_sub, _, n = S_list.shape
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_avg = S_list[0]
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + tau_list
    loss = []
    gap_tau = []
    for i in range(n_sub):
        loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
        if error_tau is True and true_tau_list is not None:
            gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        tau_list[i] = _delay_estimation(
            S_list[i], Y_avg, tau_i=tau_list_final[i], delay_max=delay_max)
        Y_list[i] = _apply_delay([Y_list[i]], [-tau_list[i]])  # XXX
    Y_avg = np.mean(Y_list, axis=0)
    tau_list %= n
    tau_list_final = tau_list_init + tau_list
    tau_list_final %= n
    for _ in range(n_iter-1):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau is True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(
                Y_list[i], new_Y_avg, tau_i=tau_list_final[i], delay_max=delay_max)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([S_list[i]], [-tau_list[i]])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
        tau_list_final = tau_list_init + tau_list
        tau_list_final %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau is True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau_with_f(
    S_list, n_iter, noise=1, use_f=True, delay_max=10, tau_list_init=None
):
    n_sub, _, n = S_list.shape
    if tau_list_init is None:
        tau_list_init = np.zeros(n_sub, dtype=int)
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    tau_list_final = tau_list_init + tau_list
    for _ in range(n_iter):
        for i in range(n_sub):
            sanity_check = 0  # XXX to be removed
            if sanity_check:
                Y_avg = np.mean(Y_list, axis=0)
                new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
                tau_list[i] += _delay_estimation(Y_list[i], new_Y_avg, delay_max=delay_max)
            else:
                tau_list[i] += _delay_estimation_with_f(
                    Y_list, i, tau_i=tau_list_final[i], noise=noise, use_f=use_f, delay_max=delay_max)
            tau_list[i] %= n
            Y_list[i] = _apply_delay_one_sub(S_list[i], -tau_list[i])
        tau_list_final = tau_list_init + tau_list
        tau_list_final %= n
    return tau_list

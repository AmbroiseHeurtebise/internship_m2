import numpy as np
# import math
# import scipy
# from scipy.ndimage import convolve1d


def _apply_delay_one_sub(Y, tau):
    return np.roll(Y, -tau, axis=1)


def _apply_delay(Y_list, tau_list):
    return np.array([_apply_delay_one_sub(Y, tau) for Y, tau in zip(Y_list, tau_list)])


def _loss_function(Y_list, Y_avg):
    loss = np.mean((Y_list - Y_avg) ** 2)
    return loss


def _loss_delay(S_list, tau_list):
    Y_list = _apply_delay(S_list, tau_list)
    Y_avg = np.mean(Y_list, axis=0)
    return _loss_function(Y_list, Y_avg)


def _loss_delay_ref(S_list, tau_list, Y_avg):
    Y_list = _apply_delay(S_list, tau_list)
    return _loss_function(Y_list, Y_avg)


# def _delay_estimation(Y, Y_avg):
#     _, n = np.shape(Y)
#     conv = np.array([convolve1d(y, y_avg[::-1], mode='wrap',
#                     origin=math.ceil(n/2)-1) for y, y_avg in zip(Y, Y_avg)])
#     conv_norm = np.sum(conv, axis=0)
#     new_tau = np.argmax(conv_norm)
#     return new_tau


def _delay_estimation(Y, Y_avg):
    conv = np.array([np.convolve(
        np.concatenate((y, y[:-1])), y_avg[::-1], mode='valid')
        for y, y_avg in zip(Y, Y_avg)])
    conv_norm = np.sum(conv, axis=0)
    new_tau = np.argmax(conv_norm)
    return new_tau


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


def _optimization_tau_approach1(S_list, n_iter, error_tau=False, true_tau_list=None):
    n_sub, _, n = S_list.shape
    Y_avg = np.mean(S_list, axis=0)
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    gap_tau = []
    for _ in range(n_iter):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau is True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(Y_list[i], new_Y_avg)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([S_list[i]], [tau_list[i]])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau is True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau_approach2(S_list, n_iter, error_tau=False, true_tau_list=None):
    n_sub, _, n = S_list.shape
    Y_avg = S_list[0]
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    gap_tau = []
    for _ in range(n_iter):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau is True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            tau_list[i] += _delay_estimation(Y_list[i], Y_avg)
            Y_list[i] = _apply_delay([S_list[i]], [tau_list[i]])
        Y_avg = np.mean(Y_list, axis=0)
        tau_list %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau is True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg


def _optimization_tau(S_list, n_iter, error_tau=False, true_tau_list=None):
    n_sub, _, n = S_list.shape
    Y_avg = S_list[0]
    Y_list = np.copy(S_list)
    tau_list = np.zeros(n_sub, dtype=int)
    loss = []
    gap_tau = []
    for i in range(n_sub):
        loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
        if error_tau is True and true_tau_list is not None:
            gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        tau_list[i] = _delay_estimation(S_list[i], Y_avg)
        Y_list[i] = _apply_delay([Y_list[i]], [tau_list[i]])  # XXX
    Y_avg = np.mean(Y_list, axis=0)
    tau_list %= n
    for _ in range(n_iter-1):
        for i in range(n_sub):
            loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
            if error_tau is True and true_tau_list is not None:
                gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
            new_Y_avg = n_sub / (n_sub - 1) * (Y_avg - Y_list[i] / n_sub)
            tau_list[i] += _delay_estimation(Y_list[i], new_Y_avg)
            old_Y = Y_list[i].copy()
            Y_list[i] = _apply_delay([S_list[i]], [tau_list[i]])
            Y_avg += (Y_list[i] - old_Y) / n_sub
        tau_list %= n
    loss.append(_loss_delay_ref(S_list, tau_list, Y_avg))
    if error_tau is True and true_tau_list is not None:
        gap_tau.append(distance_between_delays(true_tau_list, tau_list, n))
        return loss, tau_list, Y_avg, gap_tau
    return loss, tau_list, Y_avg

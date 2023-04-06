# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np
import scipy
from picard import picard
from .reduce_data import reduce_data
from .optimization_tau import _apply_delay, _apply_delay_one_sub


def permica(
    X,
    n_components=None,
    dimension_reduction="pca",
    max_iter=1000,
    random_state=None,
    tol=1e-7,
    optim_delays=False,
    delay_max=10,
):
    """
    Performs one ICA per group (ex: subject) and align sources
    using the hungarian algorithm.

    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    dimension_reduction: str, optional
        if srm: use srm to reduce the data
        if pca: use group specific pca to reduce the data
    max_iter : int, optional
        Maximum number of iterations to perform
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.

    Returns
    -------
    P : np array of shape (n_groups, n_components, n_features)
        K is the projection matrix that projects data in reduced space
    W : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    S : np array of shape (n_components, n_samples)
        Estimated source

    See also
    --------
    groupica
    multiviewica
    """
    P, X = reduce_data(
        X, n_components=n_components, dimension_reduction=dimension_reduction
    )
    n_pb, p, n = X.shape
    W = np.zeros((n_pb, p, p))
    S = np.zeros((n_pb, p, n))
    for i, x in enumerate(X):
        Ki, Wi, Si = picard(
            x,
            ortho=False,
            extended=False,
            centering=False,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        scale = np.linalg.norm(Si, axis=1)
        S[i] = Si / scale[:, None]
        W[i] = np.dot(Wi, Ki) / scale[:, None]
    tau_list = np.zeros(n_pb, dtype=int)
    if optim_delays:
        tau_list = delay_estimation_with_scale_perm(S, delay_max=delay_max)
        S = _apply_delay(S, -tau_list)
    orders, signs, S = _find_ordering(S)
    for i, (order, sign) in enumerate(zip(orders, signs)):
        W[i] = sign[:, None] * W[i][order, :]
    return P, W, S, tau_list


def _hungarian(M):
    u, order = scipy.optimize.linear_sum_assignment(-abs(M))
    vals = M[u, order]
    cost = np.sum(np.abs(vals))
    return order, np.sign(vals), cost


def delay_estimation_with_scale_perm(S_list, delay_max=10):
    """
    Finds a list of delays from a list of sources
    which are potentially permuted and have different scales.

    Parameters
    ----------
    S_list : np array of shape (n_groups, n_features, n_samples)
        The list of sources found after calling Picard algorithm.
    delay_max : int, optional (defaults to 10)
        Delays between subjects' sources will be searched
        in the segment [-delay_max, delay_max], so delay_max should be
        in the segment [0, n//2].
        If None, delays will be searched in [0, n].

    Returns
    -------
    tau_list : np array of shape(n_groups, )
        Estimated delays
    """
    m, _, n = S_list.shape
    S = S_list[0].copy()
    tau_list = np.zeros(m, dtype=int)
    if delay_max is not None:
        delays = np.concatenate((np.arange(delay_max+1), np.arange(n-delay_max, n)))
    else:
        delays = np.arange(n)
    for i, s in enumerate(S_list[1:]):
        objective = []
        for delay in delays:
            s_delayed = _apply_delay_one_sub(s, -delay)
            M = np.dot(S, s_delayed.T)
            _, _, cost = _hungarian(M)
            objective.append(cost)
        optimal_delay = np.argmax(objective)
        if delay_max is not None and optimal_delay > delay_max:
            optimal_delay += n - 2 * delay_max - 1
        tau_list[i + 1] = optimal_delay
    return tau_list


def _find_ordering(S_list, n_iter=10):
    n_pb, p, _ = S_list.shape
    for i in range(len(S_list)):
        S_list[i] /= np.linalg.norm(S_list[i], axis=1, keepdims=1)
    S = S_list[0].copy()
    order = np.arange(p)[None, :] * np.ones(n_pb, dtype=int)[:, None]
    signs = np.ones_like(order)
    for _ in range(n_iter):
        for i, s in enumerate(S_list[1:]):
            M = np.dot(S, s.T)
            order[i + 1], signs[i + 1], _ = _hungarian(M)
        S = np.zeros_like(S)
        for i, s in enumerate(S_list):
            S += signs[i][:, None] * s[order[i]]
        S /= n_pb
    return order, signs, S

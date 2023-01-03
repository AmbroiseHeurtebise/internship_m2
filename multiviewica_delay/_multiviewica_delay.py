# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np
from scipy.linalg import expm
from time import time
import warnings
from .reduce_data import reduce_data
from ._permica import permica
from ._groupica import groupica
from .optimization_tau import (
    _optimization_tau,
    _optimization_tau_approach1,  # XXX
    _optimization_tau_approach2,  # XXX
    _apply_delay_one_sub,
    _apply_delay,
    _optimization_tau_with_f
)


def multiviewica_delay(
    X,
    n_components=None,
    dimension_reduction="pca",
    noise=1.0,
    max_iter=1000,
    init="permica",
    optim_delays_permica=True,
    optim_delays_ica=True,
    delay_max=10,
    n_iter_delay=3,
    early_stopping_delay=None,
    every_N_iter_delay=1,
    optim_approach=None,  # XXX to be removed
    optim_delays_with_f=False,  # XXX to be removed
    n_iter_f=2,  # XXX to be removed
    use_f=True,  # XXX to be removed
    random_state=None,
    tol=1e-3,
    verbose=False,
    return_loss=False,
    return_basis_list=False,
    return_delays_every_iter=False,  # XXX to be removed
    return_unmixing_delays_both_phases=False,  # XXX to be removed
):
    """
    Performs MultiViewICA.
    It optimizes:
    :math:`l(W) = mean_t [sum_k log(cosh(Y_{avg}(t)[k])) + sum_i l_i(X_i(t))]`
    where
    :math:`l_i(X_i(t)) = - log(|W_i|) + 1/(2 noise) ||W_iX_i(t) - Y_{avg}(t)||^2`
    :math:`X_i` is the data of group i (ex: subject i)
    :math:`W_i` is the mixing matrix of subject i
    and
    :math:`Y_avg = mean_i W_i X_i`

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
    noise : float, optional
        Gaussian noise level
    max_iter : int, optional
        Maximum number of iterations to perform
    init : str or np array of shape (n_groups, n_components, n_components)
        If permica: initialize with perm ICA, if groupica, initialize with
        group ica. Else, use the provided array to initialize.
    optim_delays_permica : bool, optional
        Decides if we estimate delays during phase 1 or not
    optim_delays_ica : bool, optional
        Decides if we estimate delays during phase 2 or not
    delay_max : int, optional
        The maximum delay between two subjects
    n_iter_delay : int, optional
        The number of loops used in phase 2 to estimate delays
    early_stopping_delay : int, optional
        If None: estimate delays in phase 2 until convergence. Else, stop
        estimating delays from iteration number early_stopping_delay.
    every_N_iter_delay : int, optional
        Estimate delays in phase 2 every every_N_iter_delay iterations.
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.
    verbose : bool, optional
        Print information

    Returns
    -------
    P : np array of shape (n_groups, n_components, n_features)
        K is the projection matrix that projects data in reduced space
    W : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    S : np array of shape (n_components, n_samples)
        Estimated source
    tau_list : np array of shape(n_groups, )
        Estimated delays at the end of the algorithm
    tau_list_init : np array of shape(n_groups, )
        Estimated delays after the initialization

    See also
    --------
    groupica
    permica
    """
    P, X = reduce_data(
        X, n_components=n_components, dimension_reduction=dimension_reduction
    )
    # Initialization
    n_pb, _, n = X.shape
    tau_list_init = np.zeros(n_pb, dtype=int)
    if type(init) is str:
        if init not in ["permica", "groupica"]:
            raise ValueError("init should either be permica or groupica")
        if init == "permica":
            _, W, S, tau_list_init = permica(
                X, max_iter=max_iter, random_state=random_state, tol=tol,
                optim_delays=optim_delays_permica, delay_max=delay_max
            )
        else:
            _, W, S = groupica(
                X, max_iter=max_iter, random_state=random_state, tol=tol
            )
    else:
        if type(init) is not np.ndarray:
            raise TypeError("init should be a numpy array")
        W = init
    X_rescaled = _apply_delay(X, -tau_list_init)

    if return_delays_every_iter:  # XXX needs to be removed
        tau_list_main = _multiview_ica_main(
            X_rescaled,
            noise=noise,
            n_iter=max_iter,
            tol=tol,
            init=W,
            optim_delays_ica=optim_delays_ica,
            delay_max=delay_max,
            tau_list_init=tau_list_init,
            n_iter_delay=n_iter_delay,
            early_stopping_delay=early_stopping_delay,
            every_N_iter_delay=every_N_iter_delay,
            optim_delays_with_f=optim_delays_with_f,
            optim_approach=optim_approach,
            n_iter_f=n_iter_f,
            use_f=use_f,
            verbose=verbose,
            return_loss=return_loss,
            return_basis_list=return_basis_list,
            return_delays_every_iter=return_delays_every_iter,
        )
        return tau_list_init, tau_list_main

    W_init = W.copy()  # XXX to be removed

    # Performs multiview ica
    W, S, tau_list_main, loss_total = _multiview_ica_main(
        X_rescaled,
        noise=noise,
        n_iter=max_iter,
        tol=tol,
        init=W,
        optim_delays_ica=optim_delays_ica,
        delay_max=delay_max,
        tau_list_init=tau_list_init,
        n_iter_delay=n_iter_delay,
        early_stopping_delay=early_stopping_delay,
        every_N_iter_delay=every_N_iter_delay,
        optim_delays_with_f=optim_delays_with_f,
        optim_approach=optim_approach,
        n_iter_f=n_iter_f,
        use_f=use_f,
        verbose=verbose,
        return_loss=return_loss,
        return_basis_list=return_basis_list,
    )

    if return_unmixing_delays_both_phases:  # XXX to be removed
        return W_init, tau_list_init, W, tau_list_main

    tau_list = tau_list_init + tau_list_main
    tau_list %= n
    if return_loss or return_basis_list:
        return P, W, S, tau_list, loss_total
    return P, W, S, tau_list, tau_list_init  # XXX


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))


def _multiview_ica_main(
    X_list,
    noise=1.0,
    n_iter=1000,
    tol=1e-6,
    verbose=False,
    init=None,
    optim_delays_ica=True,
    delay_max=10,
    tau_list_init=None,
    n_iter_delay=3,
    early_stopping_delay=None,
    every_N_iter_delay=1,
    optim_approach=None,  # XXX to be removed
    optim_delays_with_f=False,  # XXX to be removed
    n_iter_f=2,  # XXX to be removed
    use_f=True,  # XXX to be removed
    ortho=False,
    return_gradients=False,
    timing=False,
    return_loss=False,
    return_basis_list=False,
    return_delays_every_iter=False,  # XXX to be removed
):
    n_pb, p, n = X_list.shape
    tol_init = None
    if tol > 0 and tol_init is None:
        tol_init = tol

    if tol == 0 and tol_init is None:
        tol_init = 1e-6

    if early_stopping_delay is None:
        early_stopping_delay = n_iter

    if tau_list_init is None:
        tau_list_init = np.zeros(n_pb, dtype=int)

    # Turn list into an array to make it compatible with the rest of the code
    if type(X_list) == list:
        X_list = np.array(X_list)

    # Init
    basis_list = init.copy()
    Y_avg = np.mean([np.dot(W, X) for W, X in zip(basis_list, X_list)], axis=0)
    # Start scaling
    g_norms = 0
    g_list = []
    for i in range(n_iter):
        g_norms = 0
        # Start inner loop: decrease the loss w.r.t to each W_j
        convergence = False
        for j in range(n_pb):
            X = X_list[j]
            W_old = basis_list[j].copy()
            # Y_denoise is the estimate of the sources without Y_j
            Y_denoise = Y_avg - W_old.dot(X) / n_pb
            # Perform one ICA quasi-Newton step
            converged, basis_list[j], g_norm = _noisy_ica_step(
                W_old, X, Y_denoise, noise, n_pb, ortho, scale=True
            )
            convergence = convergence or converged
            # Update the average vector (estimate of the sources)
            Y_avg += np.dot(basis_list[j] - W_old, X) / n_pb
            g_norms = max(g_norm, g_norms)

        # If line search does not converge for any subject we ll stop there
        if convergence is False:
            break

        if verbose:
            print(
                "it %d, loss = %.4e, g=%.4e"
                % (
                    i + 1,
                    _loss_total(basis_list, X_list, Y_avg, noise),
                    g_norms,
                )
            )
        if g_norms < tol_init:
            break
    # Start outer loop
    if timing:
        t0 = time()
        timings = []
    g_norms = 0
    S_list = np.array([W.dot(X) for W, X in zip(basis_list, X_list)])
    Y_list = S_list.copy()
    loss_total = []
    loss_partial = []  # XXX
    W_total = []
    # tau_list_total = np.zeros(n_pb, dtype=int)
    tau_list_every_iter = []  # XXX to be removed
    if return_loss:
        loss_total.append(_loss_total(basis_list, X_list, Y_avg, noise))
        loss_partial.append(np.mean((Y_list - np.mean(Y_list, axis=0)) ** 2))  # XXX
    for i in range(n_iter):
        # sign_list = np.ones(n_pb, p)
        tau_list = np.zeros(n_pb, dtype=int)
        if optim_delays_ica and i < early_stopping_delay and i % every_N_iter_delay == 0:
            # Delay estimation
            if optim_delays_with_f:  # XXX to be removed
                tau_list = _optimization_tau_with_f(
                    S_list, n_iter=n_iter_f, noise=noise, use_f=use_f,
                    delay_max=delay_max, tau_list_init=tau_list_init)
            else:  # XXX
                if optim_approach is None:
                    _, tau_list, Y_avg = _optimization_tau(
                        S_list, n_iter_delay, delay_max=delay_max, tau_list_init=tau_list_init)
                elif optim_approach == 1:
                    _, tau_list, Y_avg = _optimization_tau_approach1(
                        S_list, n_iter_delay, delay_max=delay_max, tau_list_init=tau_list_init)
                elif optim_approach == 2:
                    _, tau_list, Y_avg = _optimization_tau_approach2(
                        S_list, n_iter_delay, delay_max=delay_max, tau_list_init=tau_list_init)
                else:
                    raise ValueError("optim_approach should be either None, 1 or 2")
            Y_list = _apply_delay(S_list, -tau_list)
            if optim_delays_with_f:  # XXX to be removed
                Y_avg = np.mean(Y_list, axis=0)
            # tau_list_total += tau_list
            # tau_list_total %= n
            tau_list_every_iter.append(tau_list)  # XXX to be removed
        # basis_list = ...
        if return_loss:
            new_X_list = _apply_delay(X_list, -tau_list)
            loss_total.append(_loss_total(basis_list, new_X_list, Y_avg, noise))
            y_avg = np.mean(Y_list, axis=0)
            loss_partial.append(np.mean((Y_list - y_avg) ** 2))
        if return_basis_list:
            W_total.append(basis_list.copy())
        g_norms = 0
        convergence = False
        # Start inner loop: decrease the loss w.r.t to each W_j
        for j in range(n_pb):
            X = _apply_delay_one_sub(X_list[j], -tau_list[j])
            W_old = basis_list[j].copy()
            # Y_denoise is the estimate of the sources without Y_j
            Y_denoise = Y_avg - Y_list[j] / n_pb
            # Perform one ICA quasi-Newton step
            converged, basis_list[j], g_norm = _noisy_ica_step(
                W_old, X, Y_denoise, noise, n_pb, ortho
            )
            # Update the average vector (estimate of the sources)
            Y_list[j] = np.dot(basis_list[j], X)
            # Y_list[j] += _apply_delay_one_sub(np.dot(basis_list[j] - W_old, X_list[j]), -tau_list[j])
            S_list[j] = _apply_delay_one_sub(Y_list[j], tau_list[j])
            Y_avg += np.dot(basis_list[j] - W_old, X) / n_pb
            # Y_avg = np.mean(Y_list, axis=0)
            g_norms = max(g_norm, g_norms)
            convergence = converged or convergence
        if convergence is False:
            break

        g_list.append(g_norms)
        if timing:
            timings.append(
                (
                    i,
                    time() - t0,
                    _loss_total(basis_list, X_list, Y_avg, noise),
                    g_norms,
                )
            )

        if verbose:
            print(
                "it %d, loss = %.4e, g=%.4e"
                % (
                    i + 1,
                    _loss_total(basis_list, X_list, Y_avg, noise),
                    g_norms,
                )
            )
        if g_norms < tol:
            break

    else:
        warnings.warn(
            "Multiview ICA has not converged - gradient norm: %e " % g_norms
        )
    if return_gradients:
        return basis_list, Y_avg, tau_list, g_list

    if timing:
        return basis_list, Y_avg, tau_list, timings

    if return_basis_list:
        return basis_list, Y_avg, tau_list, W_total

    if return_delays_every_iter:
        return np.asarray(tau_list_every_iter)

    if return_loss:
        return basis_list, Y_avg, tau_list, [loss_total, loss_partial]
    return basis_list, Y_avg, tau_list, loss_total


def _loss_total(basis_list, X_list, Y_avg, noise):
    n_pb, p, _ = basis_list.shape
    loss = np.mean(_logcosh(Y_avg)) * p
    for i, (W, X) in enumerate(zip(basis_list, X_list)):
        Y = W.dot(X)
        loss -= np.linalg.slogdet(W)[1]
        loss += 1 / (2 * noise) * np.mean((Y - Y_avg) ** 2) * p
    return loss


def _loss_partial(W, X, Y_denoise, noise, n_pb):
    p, _ = W.shape
    Y = np.dot(W, X)
    loss = -np.linalg.slogdet(W)[1]
    loss += np.mean(_logcosh(Y / n_pb + Y_denoise)) * p
    fact = (1 - 1 / n_pb) / (2 * noise)
    loss += fact * np.mean((Y - n_pb * Y_denoise / (n_pb - 1)) ** 2) * p
    return loss


def _noisy_ica_step(
    W,
    X,
    Y_denoise,
    noise,
    n_pb,
    ortho,
    lambda_min=0.001,
    n_ls_tries=50,
    scale=False,
):
    """
    ICA minimization using quasi Newton method. Used in the inner loop.
    Returns
    -------
    converged: bool
        True if line search has converged
    new_W: np array of shape (m, p, p)
        New values for the basis
    g_norm: float
    """
    p, n = X.shape
    loss0 = _loss_partial(W, X, Y_denoise, noise, n_pb)
    Y = W.dot(X)
    Y_avg = Y / n_pb + Y_denoise

    # Compute relative gradient and Hessian
    thM = np.tanh(Y_avg)
    G = np.dot(thM, Y.T) / n / n_pb
    # print(G)
    const = 1 - 1 / n_pb
    res = Y - Y_denoise / const
    G += np.dot(res, Y.T) * const / noise / n
    G -= np.eye(p)
    if scale:
        G = np.diag(np.diag(G))
    # print(G)
    if ortho:
        G = 0.5 * (G - G.T)
    g_norm = np.max(np.abs(G))

    # These are the terms H_{ijij} of the approximated hessian
    # (approximation H2 in Pierre's thesis)
    h = np.dot((1 - thM ** 2) / n_pb ** 2 + const / noise, (Y ** 2).T,) / n

    # Regularize
    discr = np.sqrt((h - h.T) ** 2 + 4.0)
    eigenvalues = 0.5 * (h + h.T - discr)
    problematic_locs = eigenvalues < lambda_min
    np.fill_diagonal(problematic_locs, False)
    i_pb, j_pb = np.where(problematic_locs)
    h[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    # Compute Newton's direction
    det = h * h.T - 1
    direction = (h.T * G - G.T) / det
    if ortho:
        direction = 0.5 * (direction - direction.T)
    # print(direction)
    # Line search
    step = 1
    for j in range(n_ls_tries):
        if ortho:
            new_W = expm(-step * direction).dot(W)
        else:
            new_W = W - step * direction.dot(W)
        new_loss = _loss_partial(new_W, X, Y_denoise, noise, n_pb)
        if new_loss < loss0:
            return True, new_W, g_norm
        else:
            step /= 2.0
    return False, W, g_norm

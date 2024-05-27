import numpy as np
from picard import picard


def sameica(
    X,
    max_iter=1000,
    tol=1e-7,
    random_state=None,
):
    """
    Performs ICA on concatenated data.

    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_features is the number of features.
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
    W : np array of shape (n_groups, n_features, n_features)
        Estimated un-mixing matrices
    S_list : np array of shape (n_groups, n_features, n_samples)
        Estimated source

    See also
    --------
    groupica
    multiviewica_delay
    """
    m, n_comp, n = X.shape
    X_reshaped = np.concatenate(X, axis=1)
    K, W, S = picard(
        X_reshaped,
        ortho=False,
        extended=False,
        centering=False,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    scale = np.linalg.norm(S, axis=1)
    S /= scale[:, None]
    W = np.dot(W, K) / scale[:, None]
    S_list = np.swapaxes(S.reshape(n_comp, m, n), 0, 1)
    W_list = np.tile(W, (m, 1, 1))

    return W_list, S_list

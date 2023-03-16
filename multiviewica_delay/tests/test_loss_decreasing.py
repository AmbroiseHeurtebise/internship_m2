import numpy as np
import pytest
from multiviewica_delay import (
    _noisy_ica_step,
    _apply_delay,
    _apply_delay_one_sub,
    _optimization_tau_with_f
)


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))


# loss function
def _loss_total_by_source(basis_list, Y_list, Y_avg, noise=1.0):
    n_pb, p, _ = basis_list.shape
    loss = np.mean(_logcosh(Y_avg)) * p
    for i, (W, Y) in enumerate(zip(basis_list, Y_list)):
        loss -= np.linalg.slogdet(W)[1]
        loss += 1 / (2 * noise) * np.mean((Y - Y_avg) ** 2) * p
    return loss


@pytest.mark.parametrize(
    "mode", ["low", "high"])
def test_loss_decreasing(mode):
    # parameters
    if mode == "low":
        m = 2
        p = 3
        n = 4
    elif mode == "high":
        m = 10
        p = 20
        n = 100
    delay_max = 20
    seed = 0
    rng = np.random.RandomState(seed)

    # signals, unmixing matrices and sources
    X_list = rng.randn(m, p, n)
    W_list = rng.randn(m, p, p)
    S_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
    S_avg = np.mean(S_list, axis=0)

    # loss over 20 iterations
    loss0 = _loss_total_by_source(W_list, S_list, S_avg)
    # print(f"loss0 : {loss0}")
    nb_iter = 20
    loss2 = 0
    for i in range(nb_iter):
        # print(f"\nIteration {i}")
        # step 1: delay optimization
        tau_list = _optimization_tau_with_f(
            S_list,
            n_iter=3,
            delay_max=delay_max
        )
        # print("tau_list : ", tau_list)
        Y_list = _apply_delay(S_list, -tau_list)
        Y_avg = np.mean(Y_list, axis=0)
        loss1 = _loss_total_by_source(W_list, Y_list, Y_avg)
        # print(f"loss1 : {loss1}")

        if i == 0:
            assert loss0 >= loss1
        else:
            assert loss2 >= loss1

        # step 2: unmixing matrices optimization
        for i in range(m):
            W_old = W_list[i].copy()
            Y_denoise = Y_avg - Y_list[i] / m
            converged, W_list[i], g_norm = _noisy_ica_step(
                W_old, Y_list[i], Y_denoise, 1.0, m, ortho=False, X_as_input=False,
            )
            S_list[i] = np.dot(W_list[i], X_list[i])
            Y_list[i] = _apply_delay_one_sub(S_list[i], -tau_list[i])
            Y_avg = np.mean(Y_list, axis=0)
        loss2 = _loss_total_by_source(W_list, Y_list, Y_avg)
        # print(f"loss2 : {loss2}")
        assert loss1 > loss2

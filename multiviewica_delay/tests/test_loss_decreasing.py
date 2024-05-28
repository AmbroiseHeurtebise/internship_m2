import numpy as np
import pytest
from multiviewica_delay import mvica_s


@pytest.mark.parametrize(
    "mode", ["one_source", "shared_delays"])
def test_loss_decreasing(mode):
    # parameters
    if mode == "one_source":
        shared_delays = False
    elif mode == "shared_delays":
        shared_delays = True
    m = 5
    p = 4
    n = 50
    max_delay = 10
    seed = 0
    rng = np.random.RandomState(seed)
    X_list = rng.randn(m, p, n)

    # MVICAD
    _, _, _, _, loss = mvica_s(
        X_list,
        max_delay=max_delay,
        random_state=np.random.RandomState(seed),
        shared_delays=shared_delays,
        optim_delays_with_f=True,
        return_loss=True,
        verbose=True)
    loss_total, _ = loss
    loss_total = np.array(loss_total)

    # assert that the loss decreases
    assert (loss_total == np.sort(loss_total)[::-1]).all()

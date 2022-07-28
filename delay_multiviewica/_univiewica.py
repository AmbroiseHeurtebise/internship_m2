import numpy as np
from picard import picard


def univiewica(X_list, random_state=None):
    W_list = []
    for x in X_list:
        K, W, _ = picard(x, random_state=random_state)
        W_list.append(np.dot(W, K))
    return W_list

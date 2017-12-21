import matplotlib.pyplot as plt
import numpy as np

from hunse_thesis.utils import initial_w, orthogonalize


def test_orthogonalize():
    n = 10000
    din, dout = 30, 10
    rng = np.random
    T = orthogonalize(rng.normal(size=(din, dout)))
    X = rng.normal(size=(n, din))
    Y = np.dot(X, T)
    print(X.std(axis=1).mean())
    print(Y.std(axis=1).mean())

    Y2 = rng.normal(size=(n, dout))
    X2 = np.dot(Y2, T.T)
    print(X2.std(axis=1).mean())
    print(Y2.std(axis=1).mean())


def test_initial_w_ortho_angle():
    ws = [initial_w((2, 2), kind='ortho') for _ in range(100000)]

    angle0 = [np.arctan2(*w[0]) for w in ws]
    angle1 = [np.arctan2(*w[1]) for w in ws]

    plt.subplot(211)
    plt.hist(angle0, bins=51)

    plt.subplot(212)
    plt.hist(angle1, bins=51)

    plt.show()

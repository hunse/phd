"""
Examine selectivity of BP and FA neurons on MNIST
"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms
from nengo_extras.convnet import softmax
from nengo_extras.data import load_mnist, one_hot_from_labels

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, BPLearner, FALearner, FALocalLearner, FASkipLearner,
    make_flat_batch_fn, make_random_batch_fn)
from hunse_thesis.offline_learning import (
    nll_cost_on_inds, class_error_on_inds)

from hunse_thesis.utils import orthogonalize, initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

# --- problem dataset
(trainX, trainY), (testX, testY) = load_mnist('~/data/mnist.pkl.gz')
labels = np.unique(trainY)
n_labels = len(labels)

# trainX, trainY = trainX[:100], trainY[:100]  # quick training set
# trainX, trainY = trainX[:1000], trainY[:1000]  # quick training set
# trainX, trainY = trainX[:10000], trainY[:10000]  # quick training set

def preprocess(images):
    images[:] *= 2
    images[:] -= 1

preprocess(trainX), preprocess(testX)

test_set = testX, testY
# test_set = test[0][:1000], test[1][:1000]

cost = nll_cost_on_inds
error = class_error_on_inds

din = trainX.shape[1]
dout = n_labels

if 0:
    # rotational task
    rng = np.random.RandomState(5)
    def rotate(X):
        X = X.reshape(-1, 28, 28)
        rots = rng.randint(0, 4, size=X.shape[0])
        flip = rng.randint(0, 2, size=X.shape[0]).astype(bool)
        for i in range(4):
            m = rots == i
            X[m] = np.rot90(X[m], k=i, axes=(1, 2))
        X[flip] = X[flip][:, :, ::-1]

    rotate(trainX)
    rotate(testX)

# --- experiment
filename = 'angle_offline_mnist.dil'

if not os.path.exists(filename):
    seed = np.random.randint(1000000)
    # seed = 1

    rng = np.random.RandomState(seed)
    print("Seed: %d" % seed)

    # --- params
    # dhids = [200, 200]
    dhids = [500, 500]

    # eta = 0.1
    # eta = 5e-2
    # eta = 2e-2
    # eta = 1e-2
    # eta = 5e-3
    # eta = 2e-3
    eta = 1e-3

    alpha = 0

    # epochs = 1
    # epochs = 3
    # epochs = 10
    # epochs = 15
    epochs = 25

    # n_per_batch = 1
    # n_per_batch = 2
    # n_per_batch = 5
    # n_per_batch = 10
    n_per_batch = 20
    # n_per_batch = 100

    # batch_fn = make_flat_batch_fn(trainX, trainY, n_per_batch)
    batch_fn = make_random_batch_fn(trainX, trainY, n_per_batch,
                                    rng=np.random.RandomState(5))

    sizes = [din] + dhids + [dout]

    # --- initial weights
    weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
    # weights = initial_weights(sizes, kind='gaussian', scale=2e-1, rng=rng)
    print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

    tau_rc = 0.05
    # amp = 0.01
    amp = 0.025
    # amp = 0.0253

    f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp)
    # f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp, damplitude=2)
    # f, df = static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp, damplitude=2)

    x = np.linspace(-1, 1, 10001)
    print("df max: %0.3f" % df(x).max())

    genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2)
    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.4)
    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.1)
    # genB = lambda shape: initial_w(shape, kind='identity', normkind='rightmean', scale=0.1)
    Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
    Bs_direct = [genB((dout, dhid)) for dhid in dhids]

    def combine_Bs(Bs):
        Bs_combined = [Bs[-1]]
        for B in Bs[-2::-1]:
            Bs_combined.insert(0, np.dot(Bs_combined[0], B))
        return Bs_combined

    for B, Bc, Bd in zip(Bs, combine_Bs(Bs), Bs_direct):
        B *= norm(B) / norm(Bc)

    print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
    print("Bc norms: %s" % ", ".join("%0.3f" % norm(B) for B in combine_Bs(Bs)))
    print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

    get_network = lambda **kwargs: Network(
        weights, f=f, df=df, biases=None, noise=0, **kwargs)

    bp_learner = BPLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
    # bp_learner.weight_norms = []
    bp_learner.delta_norms = []

    fa_learner = FALearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='global FA')
    fa_learner.Bs = [np.array(B) for B in Bs]

    fal_learner = FALocalLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='local FA')
    fal_learner.Bs = [np.array(B) for B in Bs]

    fas_learner = FASkipLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='direct FA')
    fas_learner.Bs = [np.array(B) for B in Bs_direct]

    learners = [bp_learner]
    # learners = [fas_learner]
    # learners = [bp_learner, fas_learner]
    # learners = [bp_learner, fa_learner]
    # learners = [bp_learner, fa_learner, fal_learner, fas_learner]

    for learner in learners:
        learner.train(epochs, batch_fn, test_set=test_set)

    for learner in learners:
        print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                        for i, w in enumerate(learner.network.weights)))

# --- plots
fig = plt.figure()
rows, cols = 4, 1

filt = Alpha(100, default_dt=n_per_batch)

ax = fig.add_subplot(rows, cols, 1)
ax.set_yscale('log')
for learner in learners:
    ax.plot(filt.filtfilt(learner.batch_errors), label=learner.name)

ax = fig.add_subplot(rows, cols, 2)
ax.set_yscale('log')
for learner in learners:
    if learner.delta_norms is not None:
        ax.plot(filt.filtfilt(learner.delta_norms), label=learner.name)

ax = fig.add_subplot(rows, cols, 3)
for learner in learners:
    if getattr(learner, 'bp_angles', None) is not None:
        angles = np.array(learner.bp_angles)
        ax.plot(filt.filtfilt(angles) * (180 / np.pi), label=learner.name)
ax.set_ylim([0, 90])

ax = fig.add_subplot(rows, cols, 4)
for learner in learners:
    if getattr(learner, 'bp_angles', None) is not None:
        angles = np.array(learner.bpd_angles)
        ax.plot(filt.filtfilt(angles) * (180 / np.pi), label=learner.name)
ax.set_ylim([0, 90])

fig.tight_layout()

plt.show()

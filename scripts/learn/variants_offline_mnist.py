"""
Compare different FA variants on offline MNIST problem.
"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms
from nengo_extras.data import load_mnist

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

# --- experiment
# filename = 'variants_offline_mnist.dil'
# filename = 'variants_offline_mnist2.dil'
filename = 'variants_offline_mnist3.dil'

if not os.path.exists(filename):
    # seed = np.random.randint(1000000)
    seed = 2

    rng = np.random.RandomState(seed)
    print("Seed: %d" % seed)

    # --- params

    # dhids = [20]
    # dhids = [40]
    # dhids = [80]
    # dhids = [160]

    # dhids = [40, 40]
    # dhids = [80, 80]
    # dhids = [160, 160]
    # dhids = [320, 320]
    dhids = [500, 500]

    # eta = 0.1
    # eta = 5e-2
    # eta = 2e-2
    # eta = 1e-2
    # eta = 5e-3
    eta = 2e-3
    # eta = 1e-3

    # momentum = 0
    # momentum = 0.5
    # momentum = 0.9

    alpha = 0
    # alpha = 1e-8
    # alpha = 1e-6

    # epochs = 1
    # epochs = 3
    # epochs = 10
    epochs = 25

    # n_per_batch = 1
    # n_per_batch = 2
    # n_per_batch = 5
    # n_per_batch = 10
    n_per_batch = 20
    # n_per_batch = 100

    # batch_fn = make_flat_batch_fn(trainX, trainY, n_per_batch)
    batch_fn = make_random_batch_fn(trainX, trainY, n_per_batch, rng=np.random.RandomState(5))

    sizes = [din] + dhids + [dout]

    # --- initial weights
    weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
    print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

    tau_rc = 0.05
    # amp = 0.01
    amp = 0.025
    # amp = 0.0253

    # f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp)
    # f, df = static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp)
    # f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp, damplitude=2)
    f, df = static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp, damplitude=2)

    x = np.linspace(-1, 1, 10001)
    print("df max: %0.3f" % df(x).max())

    genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2)
    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.4)
    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.1)

    def combine_Bs(Bs):
        Bs_combined = [Bs[-1].copy()]
        for B in Bs[-2::-1]:
            Bs_combined.insert(0, np.dot(Bs_combined[0], B))
        return Bs_combined

    # Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
    # Bs_direct = [genB((dout, dhid)) for dhid in dhids]
    # for B, Bc, Bd in zip(Bs, combine_Bs(Bs), Bs_direct):
    #     B *= norm(B) / norm(Bc)

    Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
    Bnorm = norm(Bs[-1])
    for B, Bc in zip(Bs, combine_Bs(Bs)):
        B *= Bnorm / norm(Bc)
    Bs_direct = combine_Bs(Bs)

    print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
    print("Bc norms: %s" % ", ".join("%0.3f" % norm(B) for B in combine_Bs(Bs)))
    print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

    get_network = lambda **kwargs: Network(
        weights, f=f, df=df, biases=None, noise=0, **kwargs)

    bp_learner = BPLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
    bp_learner.weight_norms = []

    fa_learner = FALearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='global FA')
    fa_learner.Bs = [np.array(B) for B in Bs]

    fal_learner = FALocalLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='local FA')
    fal_learner.Bs = [np.array(B) for B in Bs]

    fas_learner = FASkipLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='direct FA')
    fas_learner.Bs = [np.array(B) for B in Bs_direct]

    # learners = [bp_learner]
    # learners = [fal_learner]
    # learners = [bp_learner, fa_learner]
    learners = [bp_learner, fa_learner, fal_learner, fas_learner]

    for learner in learners:
        learner.train(epochs, batch_fn, test_set=test_set)

    for learner in learners:
        print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                        for i, w in enumerate(learner.network.weights)))

    with open(filename, 'wb') as fh:
        dill.dump(dict(
            seed=seed, dhids=dhids, epochs=epochs, n_per_batch=n_per_batch,
            eta=eta, alpha=alpha, weights=weights, tau_rc=tau_rc, amp=amp,
            learners=learners), fh)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)

# --- plot results (cols=[train, test], traces=learners)
rows = 1
cols = 2

plt.figure(figsize=(7, 4))

# - train subplot
ax = plt.subplot(rows, cols, 1)

# filt = Alpha(3000, default_dt=n_per_batch)
filt = Alpha(10000, default_dt=n_per_batch)

for learner in learners:
    x = learner.batch_errors
    # y = x
    y = filt.filtfilt(x) if len(x) > 0 else []
    batch_inds = n_per_batch * np.arange(len(x))
    ax.semilogy(batch_inds, y, label=learner.name)

plt.ylim([5e-3, 5e-1])
# ax.set_xticklabels([])
# if col == 0:
plt.xlabel('# of examples')
plt.ylabel('train error')
# if col == cols - 1:
plt.legend(loc=1)
plt.title("Train error")

# - test subplot
plt.subplot(rows, cols, 2)

filt = Alpha(1, default_dt=1)

for learner in learners:
    x = learner.test_errors
    # y = x
    y = filt.filtfilt(x) if len(x) > 0 else []
    epoch_inds = trainX.shape[0] * np.arange(1, len(x)+1)
    plt.semilogy(epoch_inds, y, label=learner.name)

plt.ylim([1e-2, 1e-1])
plt.xlabel('# of examples')
plt.ylabel('test error')
plt.legend(loc=1)
plt.title("Test error")

sns.despine()
plt.tight_layout()

plt.savefig('variants_offline_mnist.pdf')

plt.show()

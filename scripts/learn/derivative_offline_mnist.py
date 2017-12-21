"""
Compare different derivative methods on offline MNIST problem.

Results:
- Why does softlif derivative perform better than clipped lif? They both have
  similar max. Maybe because softlif is positive even below the firing
  threshold, accounting (somewhat) for the effects of noise and possibly
  helping bump some neurons up when they're only just below threshold.
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
    Network, BPLearner, FALearner, FASkipLearner,
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

# --- learners
def test_derivative(f, df):
    get_network = lambda **kwargs: Network(
        weights, f=f, df=df, biases=None, noise=0, **kwargs)

    bp_learner = BPLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
    bp_learner.weight_norms = []

    # fa_learner = FALearner(
    #     get_network(), squared_cost, rms_error, eta=eta, alpha=alpha)
    # fa_learner.Bs = [initial_w((j, i), kind='ortho', scale=2)
    #                  for i, j in zip(dhids, dhids[1:] + [dout])]
    # fa_learner.bp_angles = []
    # # fa_learner.pbp_angles = []

    fas_learner = FASkipLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='FA')
    genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2)
    fas_learner.Bs = [genB((dout, dhid)) for dhid in dhids]

    # learners = [bp_learner, fa_learner]
    learners = [bp_learner, fas_learner]
    for learner in learners:
        learner.train(epochs, batch_fn, test_set=test_set)

    for learner in learners:
        print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                        for i, w in enumerate(learner.network.weights)))

    return learners


# --- experiment

# eta = 0.1
# eta = 5e-2
eta = 2e-2
# eta = 1e-2
# eta = 5e-3
# eta = 2e-3
# eta = 1e-3

filename = 'derivative_offline_mnist_eta=%0.3f.dil' % eta


if not os.path.exists(filename):
    # seed = None
    # seed = 9
    # seed = 435382
    seed = 535482

    seed = np.random.randint(1000000) if seed is None else seed
    rng = np.random.RandomState(seed)
    print("Seed: %d" % seed)

    # --- params
    dhids = [500, 500]

    # momentum = 0
    # momentum = 0.5
    # momentum = 0.9

    alpha = 0
    # alpha = 1e-8
    # alpha = 1e-6

    # epochs = 1
    # epochs = 3
    epochs = 10

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
    amp = 0.01

    f_dfs = [
        static_f_df('lifnone', tau_rc=tau_rc, amplitude=amp),
        static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp),
        static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp),
        static_f_df('lifclip', tau_rc=tau_rc, amplitude=amp, clip=1),
        static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp, sigma=0.146),
        # static_f_df('lifclip', tau_rc=tau_rc, amplitude=amp, clip=2),
        # static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp, sigma=0.02),
    ]
    f_df_labels = ['none', 'step', 'linearized', 'clipped', 'softlif']
    # f_df_labels = ['none', 'step', 'linearized', 'clipped', 'softlif', 'clipped', 'softlif']
    assert len(f_dfs) == len(f_df_labels)

    x = np.linspace(-1, 1, 10001)
    for i, (f, df) in enumerate(f_dfs):
        f_df_labels[i] += ' (max %0.1f)' % (df(x).max() / amp)

    results = []
    for f_df in f_dfs:
        results.append(test_derivative(*f_df))

    with open(filename, 'wb') as fh:
        dill.dump(dict(
            seed=seed, dhids=dhids, epochs=epochs, n_per_batch=n_per_batch,
            eta=eta, alpha=alpha, weights=weights, tau_rc=tau_rc, amp=amp,
            f_dfs=f_dfs, f_df_labels=f_df_labels, results=results), fh)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)

# --- plot test output (sanity check)
# rows = 1
# cols = len(results[0])

# plt.figure()

# nshow = 100
# Xshow, Yshow = Xvalid[:nshow], Yvalid[:nshow]
# plot_error_line = lambda y, z, **kwargs: plt.plot(
#     np.vstack((y[:, 0], z[:, 0])), np.vstack((y[:, 1], z[:, 1])), **kwargs)

# for col in range(cols):
#     plt.subplot(rows, cols, col + 1)
#     # plt.plot(Yshow[:, 0], Yshow[:, 1], 'k.')

#     for i, (label, color) in enumerate(zip(f_df_labels, f_df_colors)):
#         learner = results[i][col]
#         Zshow = learner.network.predict(Xshow)
#         plot_error_line(Yshow, Zshow, c=color)
#         # plt.plot(Zshow[:, 0], Zshow[:, 1], '.')

# --- plot results (rows=[train, test], cols=learners)
rows = 2
cols = len(results[0])

plt.figure(figsize=(7, 6))

# filt = Alpha(3000, default_dt=n_per_batch)
filt = Alpha(10000, default_dt=n_per_batch)
for col in range(cols):
    ax = plt.subplot(rows, cols, col + 1)

    for i, label in enumerate(f_df_labels):
        learner = results[i][col]
        x = learner.batch_errors
        # y = x
        y = filt.filtfilt(x) if len(x) > 0 else []
        batch_inds = n_per_batch * np.arange(len(x))
        ax.semilogy(batch_inds, y, label=label)

    plt.ylim([5e-3, 2e-1])
    ax.set_xticklabels([])
    if col == 0:
        plt.ylabel('train error')
    # if col == cols - 1:
    # plt.legend(loc=3)
    plt.title(learner.name)

filt = Alpha(1, default_dt=1)
for col in range(cols):
    plt.subplot(rows, cols, cols + col + 1)

    for i, label in enumerate(f_df_labels):
        learner = results[i][col]
        x = learner.test_errors
        # y = x
        y = filt.filtfilt(x) if len(x) > 0 else []
        epoch_inds = trainX.shape[0] * np.arange(1, len(x)+1)
        plt.semilogy(epoch_inds, y, label=label)

    plt.ylim([2e-2, 1.5e-1])
    plt.xlabel('# of examples')
    if col == 0:
        plt.ylabel('test error')
    if col == cols - 1:
        plt.legend(loc='best')
    # plt.title(learner.name)

sns.despine()
plt.tight_layout()

# plt.savefig('derivative_offline_mnist.pdf')
plt.savefig('derivative_offline_mnist_eta=%0.3f.pdf' % eta)

plt.show()

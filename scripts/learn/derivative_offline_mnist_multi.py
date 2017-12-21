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


def run_trial(f_dfs):
    trial_seed = rng.randint(2**30)

    # --- initial weights
    weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
    # print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

    genB = lambda shape: initial_w(
        shape, kind='ortho', normkind='rightmean', scale=0.2, rng=rng)
    directBs = [genB((dout, dhid)) for dhid in dhids]


    def test_derivative(f, df):
        # batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

        get_network = lambda **kwargs: Network(
            weights, f=f, df=df, biases=None, noise=0, **kwargs)

        bp_learner = BPLearner(
            get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
        bp_learner.weight_norms = []

        fas_learner = FASkipLearner(
            get_network(), cost, error, eta=eta, alpha=alpha, name='FA')
        fas_learner.Bs = directBs

        learners = [bp_learner, fas_learner]
        for learner in learners:
            batch_fn = make_random_batch_fn(
                trainX, trainY, n_per_batch,
                rng=np.random.RandomState(trial_seed))
            learner.train(epochs, batch_fn, test_set=test_set)

        # for learner in learners:
        #     print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
        #                     for i, w in enumerate(learner.network.weights)))

        return learners


    results = []
    for f_df in f_dfs:
        results.append(test_derivative(*f_df))


    return results


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
# test_set = testX[:1000], testY[:1000]

cost = nll_cost_on_inds
error = class_error_on_inds

din = trainX.shape[1]
dout = n_labels


# --- experiment

# eta = 0.1
# eta = 5e-2
eta = 2e-2
# eta = 1e-2
# eta = 5e-3
# eta = 2e-3
# eta = 1e-3

filename = 'derivative_offline_mnist_multi_eta=%0.3f.dil' % eta


if not os.path.exists(filename):
    # seed = np.random.randint(2**20)
    seed = 1

    rng = np.random.RandomState(seed)
    print("Seed: %d" % seed)

    # --- params
    dhids = [500, 500]

    sizes = [din] + dhids + [dout]

    # momentum = 0
    # momentum = 0.5
    # momentum = 0.9

    alpha = 0
    # alpha = 1e-8
    # alpha = 1e-6

    # n_trials = 2
    n_trials = 5

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
    # batch_fn = make_random_batch_fn(trainX, trainY, n_per_batch, rng=np.random.RandomState(5))

    # --- nonlinearity
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

    learner_names = None
    batch_errors = []  # [itrial][ifdf][ilearner][ibatch]
    test_errors = []  # [itrial][ifdf][ilearner][iepoch]
    for itrial in range(n_trials):
        results = run_trial(f_dfs)
        batch_errors.append([[learner.batch_errors for learner in learners]
                             for learners in results])
        test_errors.append([[learner.test_errors for learner in learners]
                            for learners in results])
        if learner_names is None:
            learner_names = [learner.name for learner in results[0]]

    batch_errors = np.array(batch_errors)
    test_errors = np.array(test_errors)

    with open(filename, 'wb') as fh:
        dill.dump(dict(
            seed=seed, dhids=dhids, epochs=epochs, n_per_batch=n_per_batch,
            eta=eta, alpha=alpha, tau_rc=tau_rc, amp=amp,
            f_dfs=f_dfs, f_df_labels=f_df_labels, learner_names=learner_names,
            batch_errors=batch_errors, test_errors=test_errors), fh)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)


# x = np.linspace(-1, 1, 10001)
# for i, (f, df) in enumerate(f_dfs):
#     f_df_labels[i] += ' (max %0.1f)' % (df(x).max() / amp)


# --- plot results (rows=[train, test], cols=learners)
n_trials = batch_errors.shape[0]
n_fdfs = batch_errors.shape[1]
n_learners = batch_errors.shape[2]
assert n_learners == len(learner_names)

rows = 2
cols = n_learners


plt.figure(figsize=(7, 6))

filt = Alpha(3000, default_dt=n_per_batch)
# filt = Alpha(10000, default_dt=n_per_batch)
for col in range(cols):
    ax = plt.subplot(rows, cols, col + 1)

    # error = filt.filt(batch_errors[:, :, col, :], axis=-1)
    error = filt.filtfilt(batch_errors[:, :, col, :], axis=-1)
    # batch_inds = n_per_batch * np.arange(error.shape[-1])
    batch_inds = (n_per_batch/1000.) * np.arange(error.shape[-1])
    error, batch_inds = error[..., ::10], batch_inds[::10]

    sns.tsplot(data=np.transpose(error, (0, 2, 1)),
               time=batch_inds, condition=f_df_labels,
               # err_style='unit_traces',
               legend=(col == 0))

    ax.set_ylim([5e-3, 2e-1])
    ax.set_yscale('log')
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel('train error')
    ax.set_title(learner_names[col])

filt = None
# filt = Alpha(1, default_dt=1)
for col in range(cols):
    ax = plt.subplot(rows, cols, cols + col + 1)

    error = test_errors[:, :, col, :]
    # error = filt.filtfilt(batch_errors[:, :, col, :], axis=-1)
    epoch_inds = (trainX.shape[0] / 1000.) * np.arange(1, error.shape[-1]+1)

    sns.tsplot(data=np.transpose(error, (0, 2, 1)),
               time=epoch_inds, condition=f_df_labels,
               # err_style='unit_traces',
               legend=False)

    # ax.set_ylim([2e-2, 1.5e-1])
    ax.set_ylim([1e-2, 1.1e-1])
    ax.set_yscale('log')
    ax.set_xlabel('thousands of examples')
    if col == 0:
        ax.set_ylabel('test error')
    # if col == cols - 1:
    #     plt.legend(loc='best')

sns.despine()
plt.tight_layout()

# plt.savefig('derivative_offline_mnist_multi_eta=%0.3f.pdf' % eta)
plt.savefig('derivative_offline_mnist_multi_eta=%0.3f.png' % eta, dpi=300)

plt.show()

"""
Compare different derivative methods on offline linear transform problem.

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

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, BPLearner, FALearner, FASkipLearner, make_flat_batch_fn)
from hunse_thesis.offline_learning import (
    squared_cost, rms_error)

from hunse_thesis.utils import orthogonalize, initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')


def run_trial(f_dfs):
    X = genX(n)
    Y = genY(X)
    batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

    Xvalid = genX(1000)
    Yvalid = genY(Xvalid)

    # Yrms = rms(Y, axis=1).mean()
    # Yvalidrms = rms(Yvalid, axis=1).mean()

    # --- initial weights
    weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
    # print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

    # genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.2)
    genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.3)
    directBs = [genB((dout, dhid)) for dhid in dhids]


    def test_derivative(f, df):

        get_network = lambda **kwargs: Network(
            weights, f=f, df=df, biases=None, noise=0, **kwargs)

        bp_learner = BPLearner(
            get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='BP')
        bp_learner.weight_norms = []

        fas_learner = FASkipLearner(
            get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='DFA')
        fas_learner.Bs = directBs

        learners = [bp_learner, fas_learner]
        for learner in learners:
            learner.train(1, batch_fn)

        for learner in learners:
            print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                            for i, w in enumerate(learner.network.weights)))

        return learners


    results = []
    for f_df in f_dfs:
        results.append(test_derivative(*f_df))


    return results


filename = 'derivative_offline_linear_multi.dil'

# if 1:
if not os.path.exists(filename):
    seed = 1
    # seed = np.random.randint(1000000)

    rng = np.random.RandomState(seed)
    print("Seed: %d" % seed)

    din = 30

    dhids = [80, 80]
    # dhids = [160, 160]
    # dhids = [320, 320]

    dout = 10

    sizes = [din] + dhids + [dout]

    # n_trials = 2
    # n_trials = 5
    n_trials = 10

    n_per_batch = 1
    # n_per_batch = 20

    n_per_show = 40
    assert n_per_show % n_per_batch == 0
    block_size = n_per_show // n_per_batch

    # n = 2000
    # n = 9000
    # n = 12000
    # n = 15000
    n = 21000

    alpha = 0

    # --- problem dataset
    T = orthogonalize(rng.normal(size=(din, dout)))
    genX = lambda n: rng.normal(scale=1., size=(n, din))
    genY = lambda X: np.dot(X, T)

    Xref = genX(1000)
    Yref = genY(Xref)
    Yrms = rms(Yref, axis=1).mean()

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

    etas = [0.02, 0.05, 0.1, 0.2]

    learner_names = None
    errors = {}  # [eta][itrial][ifdf][ilearner]
    for eta in etas:
        errors[eta] = []
        for itrial in range(n_trials):
            results = run_trial(f_dfs)
            errors[eta].append(np.array([[
                np.reshape(learner.batch_errors, (-1, block_size)).mean(axis=1)
                for learner in learners] for learners in results]))
            if learner_names is None:
                learner_names = [learner.name for learner in results[0]]

    with open(filename, 'wb') as fh:
        dill.dump(dict(
            seed=seed, dhids=dhids, n=n, n_per_batch=n_per_batch, n_per_show=n_per_show,
            T=T, Yrms=Yrms, etas=etas, alpha=alpha,
            tau_rc=tau_rc, amp=amp, f_dfs=f_dfs, f_df_labels=f_df_labels,
            learner_names=learner_names, errors=errors), fh)
    print("Saved %r" % filename)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)
    if len(f_df_labels) == 5:
        f_df_labels = ['none', 'step', 'linearized', 'clipped', 'softlif']
    for i, learner_name in enumerate(learner_names):
        if learner_name == 'FA':
            learner_names[i] = 'DFA'

n_per_show = 40

x = np.linspace(-1, 1, 10001)
for i, (f, df) in enumerate(f_dfs):
    f_df_labels[i] += ' (max %0.1f)' % (df(x).max() / amp)

# --- plot results
errors0 = list(errors.values())[0]
n_trials = len(errors0)
n_fdfs = errors0[0].shape[0]
assert len(f_dfs) == len(f_df_labels) == n_fdfs
assert len(learner_names) == errors0[0].shape[1]

rows = len(etas)
cols = len(learner_names)

# filt = Alpha(30, default_dt=n_per_batch)
filt = Alpha(80, default_dt=n_per_show)
# filt = Alpha(100, default_dt=n_per_show)

plt.figure(figsize=(6, 8))
for row in range(rows):  # for each eta
    for col in range(cols):  # for each learner
        print("Plotting (%d, %d)" % (row, col))
        ax = plt.subplot(rows, cols, row*cols + col + 1)
        eta = etas[row]

        # (n_trials x n_fdfs x n) matrix of errors
        # print(type(errors[eta][0]))
        # print(errors[eta][0].shape)
        error = np.array([[errors[eta][itrial][ifdf, col]
                           for ifdf in range(n_fdfs)]
                          for itrial in range(n_trials)])
        error /= Yrms
        # error[~np.isfinite(error)] = 2
        error[~np.isfinite(error)] = 1e6
        error = filt.filt(error, axis=-1)
        # error = filt.filtfilt(error, axis=-1)

        batch_inds = (n_per_show/1000.) * np.arange(error.shape[-1])

        sns.tsplot(data=np.transpose(error, (0, 2, 1)),
                   time=batch_inds, condition=f_df_labels,
                   # err_style='unit_traces',
                   legend=(row == 0 and col == 0))

        ax.set_ylim([0.09, 1.2])
        ax.set_yscale('log')
        if row+1 == rows:
            ax.set_xlabel('thousands of examples')

        # # plt.xticks([0] + [1000*i for i in range(1, n // 1000 + 1, 2)])
        # plt.xticks([3000*i for i in range(0, n // 3000 + 1)])
        # plt.xlabel('# of examples')
        # plt.ylabel('train error')
        # if col+1 == cols:
        #     plt.legend(loc='best')
        # if row == 0:
        plt.title('%s ($\\eta = %0.2f$)' % (learner_names[col], eta))


sns.despine()
plt.tight_layout()
plt.savefig('derivative_offline_linear_multi.pdf')

plt.show()

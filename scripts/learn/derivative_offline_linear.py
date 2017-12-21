"""
Compare different derivative methods on offline linear transform problem.

Results:
- Why does softlif derivative perform better than clipped lif? They both have
  similar max. Maybe because softlif is positive even below the firing
  threshold, accounting (somewhat) for the effects of noise and possibly
  helping bump some neurons up when they're only just below threshold.
"""
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

n_per_batch = 1

# n = 9000
# n = 12000
# n = 15000
n = 21000

# eta = 0.4
# eta = 0.3
# eta = 0.25
# eta = 0.2
# eta = 0.1
# eta = 0.05
eta = 0.02
# eta = 0.01


alpha = 0

# --- problem dataset
T = orthogonalize(rng.normal(size=(din, dout)))
genX = lambda n: rng.normal(scale=1., size=(n, din))
genY = lambda X: np.dot(X, T)

X = genX(n)
Y = genY(X)
batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

Xvalid = genX(1000)
Yvalid = genY(Xvalid)

Yrms = rms(Y, axis=1).mean()
Yvalidrms = rms(Yvalid, axis=1).mean()

# --- initial weights
weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

# genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.2)
genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.3)
directBs = [genB((dout, dhid)) for dhid in dhids]

# --- learners
def test_derivative(f, df):
    get_network = lambda **kwargs: Network(
        weights, f=f, df=df, biases=None, noise=0, **kwargs)

    bp_learner = BPLearner(
        get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='BP')
    bp_learner.weight_norms = []

    fas_learner = FASkipLearner(
        get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='FA')
    fas_learner.Bs = directBs

    learners = [bp_learner, fas_learner]
    for learner in learners:
        learner.train(1, batch_fn)

    for learner in learners:
        print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                        for i, w in enumerate(learner.network.weights)))

    return learners


# --- nonlinearity
tau_rc = 0.05
amp = 0.01

f_dfs = [
    static_f_df('lifnone', tau_rc=tau_rc, amplitude=amp),
    static_f_df('lifact', tau_rc=tau_rc, amplitude=amp, clip=1),
    static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp),
    static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp),
    static_f_df('lifclip', tau_rc=tau_rc, amplitude=amp, clip=1),
    static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp, sigma=0.146),
    # static_f_df('lifnone', tau_rc=tau_rc, amplitude=amp, damplitude=2),
    # static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp, damplitude=2),
    # static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp, damplitude=2),
    # static_f_df('lifclip', tau_rc=tau_rc, amplitude=amp, clip=2),
    # static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp),
]
# f_df_labels = ['none', 'step', 'linearized', 'clipped', 'softlif']
f_df_labels = ['none', 'act', 'step', 'linearized', 'clipped', 'softlif']
# f_df_colors = ['k', 'b', 'g', 'r', 'm', 'y']
assert len(f_dfs) == len(f_df_labels)

x = np.linspace(-1, 1, 10001)
for i, (f, df) in enumerate(f_dfs):
    f_df_labels[i] += ' (max %0.1f)' % (df(x).max() / amp)
print(f_df_labels)

results = []
for f_df in f_dfs:
    results.append(test_derivative(*f_df))

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
rows = 1
cols = len(results[0])

# filt = Alpha(30, default_dt=n_per_batch)
filt = Alpha(100, default_dt=n_per_batch)

plt.figure(figsize=(6.4, 3))
for col in range(cols):
    plt.subplot(rows, cols, col + 1)

    for i, label in enumerate(f_df_labels):
        learner = results[i][col]
        x = learner.batch_errors / Yrms
        x[~np.isfinite(x)] = 1e8
        y = filt.filtfilt(x) if len(x) > 0 else []
        batch_inds = n_per_batch * np.arange(len(x))
        plt.semilogy(batch_inds, y, label=label)

    # plt.ylim([0.05, 1])
    plt.ylim([0.09, 1.2])
    # plt.xticks([0] + [1000*i for i in range(1, n // 1000 + 1, 2)])
    plt.xticks([3000*i for i in range(0, n // 3000 + 1)])
    plt.xlabel('# of examples')
    plt.ylabel('train error')
    if col+1 == cols:
        plt.legend(loc='best')
    plt.title(learner.name)

# for col in range(cols):
#     plt.subplot(rows, cols, cols + col + 1)

#     for i, label in enumerate(f_df_labels):
#         learner = results[i][col]
#         x = learner.batch_errors

#         filt = Lowpass(10, default_dt=n_per_batch)
#         y = filt.filtfilt(x) if len(x) > 0 else []
#         batch_inds = n_per_batch * np.arange(len(x))
#         plt.semilogy(batch_inds, y, label=label)

#     plt.ylabel('test error')

#     plt.title(learner.name)

sns.despine()
plt.tight_layout()
plt.savefig('derivative_offline_linear_eta=%0.3f.pdf' % eta)

plt.show()

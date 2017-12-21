"""
Get the Lillicrap et al random backprop method working on their simple example.

NOTES:
- Feedback alignment (FA) does not seem to decrease as quickly as backprop (BP)
  in general over long times. In the simple example, FA appears to do well,
  but only because the feedback matrix B is much larger magnitude than W.T,
  meaning that FA essentially has a larger learning rate than BP.
- Increasing the learning rate of BP (eta_bp) makes performance better than FA
  (in accord with the previous comment).
- Using orthogonal B helps alignment with backprop

- Initial learning plateaus seem largely related to initial weight magnitudes.
  Perhaps how these relate to learning rates?
"""
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np

from nengo.synapses import Lowpass
from nengo.utils.numpy import norm, rms

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, ShallowLearner, BPLearner, FALearner, FASkipLearner, DTPLearner,
    make_flat_batch_fn)
from hunse_thesis.offline_learning import (
    squared_cost, rms_error)

from hunse_thesis.utils import orthogonalize, initial_weights, initial_w, lsuv


# rng = np.random.RandomState(9)
# rng = np.random.RandomState(8)
rng = np.random

din = 30
# dhids = [20]
# dhids = [40]
# dhids = [80]
# dhids = [160]

# dhids = [40, 40]
dhids = [80, 80]
# dhids = [160, 160]
# dhids = [320, 320]

dout = 10
# dout = 2

sizes = [din] + dhids + [dout]

n_per_batch = 1
# n_per_batch = 2
# n_per_batch = 5
# n_per_batch = 10
# n_per_batch = 20

# n = 300
# n = 3000
n = 6000
# n = 9000
# n = 12000
# n = 15000
# n = 18000
# n = 25000
# n = 35000
# n = 50000
# n = 80000

n_batches = n // n_per_batch

# eta = 1.
# eta = 0.5
# eta = 0.25
# eta = 0.125
eta = 5e-2
# eta = 2e-2
# eta = 1e-2
# eta = 5e-3
# eta = 2e-3
# eta = 1e-3
# eta = 2e-4
# eta = 1e-4
# eta = 5e-5
# eta = 3e-5
# eta = 2e-5
# eta = 1e-5

# eta = 5e-2

# eta = [0.0057, 0.00026, 1.17e-7]

momentum = 0
# momentum = 0.5

alpha = 0
# alpha = 1e-8
# alpha = 1e-6

# --- problem dataset
T = orthogonalize(rng.normal(size=(din, dout)))
# genX = lambda n: rng.normal(scale=1., size=(n, din))
genX = lambda n: rng.normal(scale=0.5, size=(n, din))
if 1:
    # linear
    genY = lambda X: np.dot(X, T)
else:
    # genY = lambda X: np.tanh(np.dot(X, T))
    genY = lambda X: np.dot(np.tanh(X), T)

X = genX(n)
Y = genY(X)
batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

Xvalid = genX(1000)
Yvalid = genY(Xvalid)

Yrms = rms(Y, axis=1).mean()
Yvalidrms = rms(Yvalid, axis=1).mean()

# --- nonlinearity
# f, df = static_f_df('sigmoid')
# f, df = static_f_df('softlif', tau_rc=0.02)
# f, df = static_f_df('softlif', tau_rc=0.05)
# f, df = static_f_df('softlif', tau_rc=0.05, amplitude=1./30)
# f, df = static_f_df('softlif', tau_rc=0.05, amplitude=1./10)
# f, df = static_f_df('softlif', tau_rc=0.05, amplitude=1./7)
# f, df = static_f_df('softlif', tau_rc=0.05, amplitude=1./4.3)
# f, df = static_f_df('softlif', tau_rc=0.05, amplitude=1)
# f, df = static_f_df('liflinear', tau_rc=0.05, amplitude=1)
# f, df = static_f_df('liflinear', tau_rc=0.05, amplitude=1./50)
# f, df = static_f_df('liflinear', tau_rc=0.05, amplitude=0.005)

f, df = static_f_df('liflinear', tau_rc=0.05, amplitude=0.01)
# f, df = static_f_df('liflinear', tau_rc=0.05, amplitude=0.004)


# --- initial weights
# weights = initial_weights(sizes, kind='uniform', scale=0.01, rng=rng)
# weights = initial_weights(sizes, kind='uniform', scale=0.2, rng=rng)
# weights = initial_weights(sizes, kind='ortho', scale=0.003, rng=rng)
# weights = initial_weights(sizes, kind='ortho', scale=0.06, rng=rng)
# weights = initial_weights(sizes, kind='ortho', scale=0.3, rng=rng)
# weights = initial_weights(sizes, kind='ortho', rng=rng)
# weights = initial_weights(sizes, kind='ortho', scale=2, rng=rng)

weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
# weights = initial_weights(sizes, kind='gaussian', scale=3e-2, rng=rng)
# weights = initial_weights(sizes, kind='gaussian', scale=3e-3, rng=rng)

# weights = initial_weights(sizes, kind='uniform', scale=1.7e-2, rng=rng)

# lsuv(X, weights, f, verbose=1)
# lsuv(X, weights, f, target_input=True, target_std=5, verbose=1)
# lsuv(X, weights, f, target_input=True, target_std=2, verbose=1)

print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

# --- network
noise = 0
# noise = 1.
get_network = lambda **kwargs: Network(
    weights, f=f, df=df, biases=None, noise=noise, **kwargs)

network0 = get_network()
acts0, outs0 = network0.forward(X)
# acts0, outs0 = network0.forward(X[:1000])
print("Initial acts: %s" % "; ".join(
    ["%d = %0.3f, %0.3f" % (i, x.mean(), x.std(1).mean())
     for i, x in enumerate(acts0)]))
print("Initial outs: %s" % "; ".join(
    ["%d = %0.3f, %0.3f" % (i, x.mean(), x.std(1).mean())
     for i, x in enumerate(outs0)]))

if 0:
    plt.figure(101)
    for k, act in enumerate(acts0):
        plt.subplot(len(acts0), 1, k+1)
        plt.hist(act.ravel(), bins=31)
        # print(act.std(1).mean())

    plt.show()
    assert 0

# --- learners
shallow_learner = ShallowLearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='Shallow')
bp_learner = BPLearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='BP')
bp_learner.weight_norms = []

fa_learner = FALearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha)
fa_learner.Bs = [initial_w((j, i), kind='ortho', scale=2)
                 for i, j in zip(dhids, dhids[1:] + [dout])]
fa_learner.bp_angles = []
# fa_learner.pbp_angles = []

fas_learner = FASkipLearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='Our model')
# fas_learner.Bs = [initial_w((dout, dhid), kind='ortho', scale=2) for dhid in dhids]
fas_learner.Bs = [initial_w((dout, dhid), kind='ortho', normkind='rightmean') for dhid in dhids]

# V0init = rng.uniform(-1, 1, size=(dhid, din))
# Vinit = rng.uniform(-1, 1, size=(dout, dhid))
# V0init = orthogonalize(rng.uniform(-1, 1, size=(dhid, din)))
# Vinit = orthogonalize(rng.uniform(-1, 1, size=(dout, dhid)))
# V0init = np.linalg.pinv(W0init)
# Vinit = np.linalg.pinv(Winit)

bweights = initial_weights(sizes[::-1], kind='ortho', scale=2)[::-1]

dtp_learner = DTPLearner(
    get_network(backweights=bweights),
    squared_cost, rms_error, eta=eta, alpha=alpha, momentum=momentum)
# dtp_learner.eta_hat = lambda t: np.clip(0.5*t, 0.01, 1.)
dtp_learner.eta_hat = lambda t: np.clip(2*t, 0.5, 1.)
# dtp_learner.eta_hat = lambda t: np.clip(5*t, 0.5, 1.)
# dtp_learner.eta_hat = 1.
# dtp_learner.eta_hat = 0.5

# learners = [bp_learner]
# learners = [shallow_learner, bp_learner]
learners = [shallow_learner, bp_learner, fas_learner]
# learners = [shallow_learner, bp_learner, fa_learner, fas_learner]
# learners = [shallow_learner, bp_learner, fa_learner, fas_learner, dtp_learner]
# learners = [dtp_learner]

for learner in learners:
    learner.train(1, batch_fn)

# --- save results
dlearners = [
    dict(name=learner.name, batch_errors=learner.batch_errors)
    for learner in learners]

if 1:
    s_sizes = '-'.join('%d' % s for s in [din] + dhids + [dout])
    s_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'results/offline_linear_%s_eta=%0.1e_%s.npz' % (
        s_sizes, eta, s_now)

    keys = ['eta', 'n_per_batch', 'n', 'n_batches', 'din', 'dhids', 'dout',
            'T', 'X', 'Y', 'Xvalid', 'Yvalid']
    data = dict((k, globals()[k]) for k in keys)
    data['learners'] = dlearners

    np.savez(filename, **data)
    print("Saved %r" % filename)


def print_weights(learner, prefix=None):
    prefix = type(learner).__name__ if prefix is None else prefix
    print("%s: %s" % (prefix, ", ".join(
        ["||W%d|| = %0.3f" % (i, norm(w))
         for i, w in enumerate(learner.network.weights)])))

def print_acts(learner, prefix=None):
    _, outs = learner.network.forward(X[-100:])
    prefix = type(learner).__name__ if prefix is None else prefix
    print("%s: %s" % (prefix, "; ".join(
        ["%d = %0.3f, %0.3f" % (i, x.mean(), x.std(1).mean())
         for i, x in enumerate(outs)])))

def print_error(learner, prefix=None):
    prefix = type(learner).__name__ if prefix is None else prefix
    z = learner.network.predict(Xvalid)
    error_v = rms(z - Yvalid, axis=1).mean() / Yvalidrms
    print("%s: error_v = %0.3f" % (prefix, error_v))


for learner in learners:
    print_weights(learner)
    print_acts(learner)
    print_error(learner)

if 1:
    from plot_offline_linear import cosyne_plot
    cosyne_plot(dlearners, n_per_batch, Y)
    plt.show()
    sys.exit()


# plt.ion()
plt.figure(1)
plt.clf()

plt.subplot(211)
def plot_batches(x, label=None, color=None):
    filt = Lowpass(10, default_dt=n_per_batch)
    y = filt.filtfilt(x) if len(x) > 0 else []
    batch_inds = n_per_batch * np.arange(len(x))
    plt.semilogy(batch_inds, y, label=label, color=color)

plot_batches(shallow_learner.batch_errors, 'Sh', 'k')
plot_batches(bp_learner.batch_errors, 'BP')
plot_batches(fa_learner.batch_errors, 'FA')
plot_batches(fas_learner.batch_errors, 'FASkip')
plot_batches(dtp_learner.batch_errors, 'DTP')
plt.ylabel('RMS error')
plt.legend(loc=1)

plt.subplot(212)
def plot_angles(x, label=None):
    filt = Lowpass(10, default_dt=n_per_batch)
    y = filt.filtfilt(x) if len(x) > 0 else []
    batch_inds = n_per_batch * np.arange(len(x))
    plt.plot(batch_inds, y, label=label)

plot_angles(np.array(fa_learner.bp_angles) * (180 / np.pi), label='FA $\measuredangle$ BP')
# plot_angles(np.array(fa_learner.pbp_angles) * (180 / np.pi), label='FA $\measuredangle$ PBP')
plt.ylim([0, 90])
plt.xlabel('training examples')
plt.ylabel('$\Delta_h$ angle (degrees)')
plt.legend(loc=1)

# plt.figure(11)
# def plot_weight_norms(x, label=None):
#     x = [] if x is None else x
#     batch_inds = n_per_batch * np.arange(len(x))
#     plt.plot(batch_inds, x, label=label)

# plot_weight_norms(bp_learner.weight_norms, 'BP')

# plt.figure(2)
# plt.clf()

# plt.semilogy(example_inds, errors_bp, label='backprop')
# plt.semilogy(example_inds, errors_fa, label='feedback alignment')
# plt.legend(loc='best')
# plt.xlabel('training examples')
# plt.ylabel('RMS error')

# plt.savefig('feedback-alignment-%d.pdf' % n)

# --- plot of testing points
if 0:
    plt.figure(3)

    def plot_errors(learner, c='b'):
        n = 50
        x, y = Xvalid[:n], Yvalid[:n]
        z = learner.network.predict(x)
        plt.plot(np.vstack((y[:, 0], z[:, 0])), np.vstack((y[:, 1], z[:, 1])), c)

    # plot_errors(shallow_learner, 'k')
    plot_errors(bp_learner, 'b')
    # plot_errors(fa_learner, 'g')
    # plot_errors(dtp_learner, 'r')

plt.show()

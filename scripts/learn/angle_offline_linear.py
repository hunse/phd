"""
Look at the angle between FA and BP updates.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, BPLearner, FALearner, FALocalLearner, FASkipLearner, make_flat_batch_fn)
from hunse_thesis.offline_learning import (
    squared_cost, rms_error)

from hunse_thesis.utils import orthogonalize, initial_weights, initial_w


def WBeigs(learner):
    # --- eigenvalues of W^T B
    W = learner.network.weights[-1]
    B = learner.Bs[-1]
    WB = np.dot(B, W)

    e, V = np.linalg.eig(WB)
    return e.real


sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

seed = 1
# seed = np.random.randint(1000000)

rng = np.random.RandomState(seed)
print("Seed: %d" % seed)

din = 30

# dhids = [80, 80]
# dhids = [160, 160]
dhids = [320, 320]

dout = 10

sizes = [din] + dhids + [dout]

n_per_batch = 1

# n = 3
# n = 1000
# n = 9000
n = 12000
# n = 15000
# n = 21000

# eta = 0.4
# eta = 0.3
# eta = 0.25
# eta = 0.2
# eta = 0.1
# eta = 0.05
# eta = 0.02
eta = 0.01
# eta = 0.002

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

# Bnorm = 1.
genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.2)
# genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.3)
# genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=1.0)
# Bs_direct = [genB((dout, dhid)) for dhid in dhids]

def combine_Bs(Bs):
    Bs_combined = [Bs[-1].copy()]
    for B in Bs[-2::-1]:
        Bs_combined.insert(0, np.dot(Bs_combined[0], B))
    return Bs_combined

Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
Bnorm = norm(Bs[-1])
for B, Bc in zip(Bs, combine_Bs(Bs)):
    B *= Bnorm / norm(Bc)

Bs_direct = combine_Bs(Bs)
# print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
# print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

# --- nonlinearity
tau_rc = 0.05
# amp = 0.01
amp = 0.025

# f, df = static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp)
f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp)

# --- learners
get_network = lambda **kwargs: Network(
    weights, f=f, df=df, biases=None, noise=0, **kwargs)

bp_learner = BPLearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='BP')
# bp_learner.weight_norms = []
bp_learner.delta_norms = []

fa_learner = FALearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='FA')
fa_learner.Bs = Bs
fa_learner.delta_norms = []
fa_learner.bp_angles = []
fa_learner.bpd_angles = []

# fal_learner = FALocalLearner(  # Identical to FASkipLearner
#     get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='LFA')
# fal_learner.Bs = Bs
# fal_learner.delta_norms = []
# fal_learner.bp_angles = []
# fal_learner.bpd_angles = []

fas_learner = FASkipLearner(
    get_network(), squared_cost, rms_error, eta=eta, alpha=alpha, name='DFA')
fas_learner.Bs = Bs_direct
fas_learner.delta_norms = []
fas_learner.bpd_angles = []
fas_learner.bpu_angles = []

e = WBeigs(fas_learner)
print("Eigs0: %0.3f, %0.3f [%0.3f, %0.3f]" % (e.mean(), e.std(), e.min(), e.max()))

# learners = [bp_learner, fa_learner]
learners = [fas_learner]
# learners = [bp_learner, fa_learner, fas_learner]
for learner in learners:
    # learner.train(1, batch_fn)
    learner.train(1, batch_fn, test_set=(Xvalid, Yvalid))

# for learner in learners:
#     print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
#                     for i, w in enumerate(learner.network.weights)))

e = WBeigs(fas_learner)
print("Eigs1: %0.3f, %0.3f [%0.3f, %0.3f]" % (e.mean(), e.std(), e.min(), e.max()))

# --- plots
n_batches = len(learners[0].batch_errors)
n_hids = len(dhids)
batch_inds = (n_per_batch / 1000.) * np.arange(n_batches)
# epoch_inds = (trainX.shape[0] / 1000.) * np.arange(1, epochs+1)

fig = plt.figure()
rows, cols = 5, 1
# layer_styles = ('-', '-.', ':', '--')
layer_styles = ('-', '--', ':')

# filt = Alpha(10, default_dt=n_per_batch)
# filt = Alpha(20, default_dt=n_per_batch)
# filt = Alpha(30, default_dt=n_per_batch)
filt = Alpha(100, default_dt=n_per_batch)

ax = fig.add_subplot(rows, cols, 1)
ax.set_yscale('log')
batch_errors = np.array([learner.batch_errors for learner in learners])
ax.plot(batch_inds, filt.filtfilt(batch_errors, axis=1).T)

ax = fig.add_subplot(rows, cols, 2)
ax.set_yscale('log')
delta_norms = np.array([
    x.delta_norms if x.delta_norms else np.nan*np.ones((n_batches, n_hids))
    for x in learners])
delta_norms = filt.filtfilt(delta_norms, axis=1)
for i in range(n_hids):
    ax.set_color_cycle(None)
    ax.plot(batch_inds, delta_norms[..., i].T, linestyle=layer_styles[i])

ax = fig.add_subplot(rows, cols, 3)
bp_angles = (180 / np.pi) * np.array([
    getattr(learner, 'bp_angles', np.nan*np.ones((n_batches, n_hids)))
    for learner in learners])
bp_angles = filt.filtfilt(bp_angles, axis=1)
for i in range(n_hids):
    ax.set_color_cycle(None)
    ax.plot(batch_inds, bp_angles[..., i].T, linestyle=layer_styles[i])
ax.set_ylim([0, 90])

ax = fig.add_subplot(rows, cols, 4)
bpd_angles = (180 / np.pi) * np.array([
    getattr(learner, 'bpd_angles', np.nan*np.ones((n_batches, n_hids)))
    for learner in learners])
bpd_angles = filt.filtfilt(bpd_angles, axis=1)
for i in range(n_hids):
    ax.set_color_cycle(None)
    ax.plot(batch_inds, bpd_angles[..., i].T, linestyle=layer_styles[i])
ax.set_ylim([0, 90])

ax = fig.add_subplot(rows, cols, 5)
bpu_angles = (180 / np.pi) * np.array([
    getattr(learner, 'bpu_angles', np.nan*np.ones((n_batches, n_hids)))
    for learner in learners])
bpu_angles = filt.filtfilt(bpu_angles, axis=1)
for i in range(n_hids):
    ax.set_color_cycle(None)
    ax.plot(batch_inds, bpu_angles[..., i].T, linestyle=layer_styles[i])
ax.set_ylim([0, 90])

fig.tight_layout()

plt.show()

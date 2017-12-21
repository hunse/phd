"""
Compare different FA variants on offline linear transform problem.

Specifically, we'll compare around the critical learning rate,
where increasing the LR causes instability.
"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms
from nengo_extras.data import load_mnist

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, BPLearner, BPLocalLearner, FALearner, FALocalLearner, FASkipLearner,
    make_flat_batch_fn)
from hunse_thesis.offline_learning import (
    squared_cost, rms_error)

from hunse_thesis.utils import orthogonalize, initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

# seed = None
seed = 1

seed = np.random.randint(1000000) if seed is None else seed
rng = np.random.RandomState(seed)
print("Seed: %d" % seed)

din = 30

# dhids = [40, 40]
# dhids = [80, 80]
dhids = [160, 160]
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
n = 5000
# n = 6000
# n = 9000
# n = 12000

epochs = 1

etas = [4.0e-2, 4.7e-2, 5.0e-2, 6.0e-2]

alpha = 0

# --- problem dataset
T = orthogonalize(rng.normal(size=(din, dout)))
genX = lambda n: rng.normal(scale=1., size=(n, din))
# genX = lambda n: rng.normal(scale=0.5, size=(n, din))
genY = lambda X: np.dot(X, T)

X = genX(n)
Y = genY(X)
batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

Xtest = genX(1000)
Ytest = genY(Xtest)
test_set = Xtest, Ytest

Yrms = rms(Y, axis=1).mean()
Ytestrms = rms(Ytest, axis=1).mean()

cost = squared_cost
error = rms_error

# --- neuron
tau_rc = 0.05
# amp = 0.01
amp = 0.025
# amp = 0.0253

# f, df = static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp)
# f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp, damplitude=2)
f, df = static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp, damplitude=2)

x = np.linspace(-1, 1, 10001)
print("df max: %0.3f" % df(x).max())

# --- Experiment
n_trials = 5
# n_trials = 1

trial_weights = []
for i_trial in range(n_trials):
    # --- initial weights
    # weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
    weights = initial_weights(sizes, kind='gaussian', scale=5e-2, rng=rng)
    # weights = initial_weights(sizes, kind='gaussian', scale=1e-2, rng=rng)

    # print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

    genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2)
    Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
    Bs_direct = [genB((dout, dhid)) for dhid in dhids]

    def combine_Bs(Bs):
        Bs_combined = [Bs[-1]]
        for B in Bs[-2::-1]:
            Bs_combined.insert(0, np.dot(Bs_combined[0], B))
        return Bs_combined

    for Bd, Bc in zip(Bs_direct, combine_Bs(Bs)):
        Bd *= norm(Bc) / norm(Bd)

    trial_weights.append((weights, Bs, Bs_direct))

eta_errors = []
for eta in etas:
    trial_errors = []
    for i_trial, (weights, Bs, Bs_direct) in enumerate(trial_weights):
        get_network = lambda **kwargs: Network(
            weights, f=f, df=df, biases=None, noise=0, **kwargs)

        fa_learner = FALearner(
            get_network(), cost, error, eta=eta, alpha=alpha, name='global FA')
        fa_learner.Bs = [np.array(B) for B in Bs]

        fal_learner = FALocalLearner(
            get_network(), cost, error, eta=eta, alpha=alpha, name='local FA')
        fal_learner.Bs = [np.array(B) for B in Bs]

        fas_learner = FASkipLearner(
            get_network(), cost, error, eta=eta, alpha=alpha, name='direct FA')
        fas_learner.Bs = [np.array(B) for B in Bs_direct]

        # learners = [fa_learner]
        learners = [fa_learner, fal_learner, fas_learner]

        trials_i = []
        for learner in learners:
            learner.train(epochs, batch_fn, test_set=test_set)
            trials_i.append(learner.batch_errors)

        trial_errors.append(trials_i)

    eta_errors.append(trial_errors)

learner_names = [learner.name for learner in learners]
# trial_errors = np.array(trial_errors)
eta_errors = np.array(eta_errors)
batch_inds = n_per_batch * np.arange(eta_errors.shape[-1])

# --- plot results (traces=learners)
plt.figure(figsize=(6.35, 6.0))

rows, cols = 2, 2
assert len(etas) <= rows*cols

for i, eta in enumerate(etas):
    trial_errors = eta_errors[i]

    filt = Alpha(30, default_dt=n_per_batch)
    trial_errors[~np.isfinite(trial_errors)] = 1e6
    trial_errors = trial_errors.clip(None, 1e6)
    trial_errors = filt.filt(trial_errors / Yrms, axis=-1)
    # trial_errors = filt.filtfilt(trial_errors / Yrms, axis=-1)

    ax = plt.subplot(rows, cols, i+1)
    # ax = sns.tsplot(data=np.transpose(trial_errors, (0, 2, 1)),
    #                 time=batch_inds, condition=learner_names)
    sns.tsplot(data=np.transpose(trial_errors, (0, 2, 1)),
               time=batch_inds, condition=learner_names, err_style='unit_traces',
               legend=(i == 0))

    # ax.set(yscale='log')
    plt.ylim([1e-1, 1.2e0])
    plt.xlabel('# of examples')
    plt.ylabel('normalized RMS error')
    plt.title('$\eta$ = %0.1e' % eta)

    sns.despine()

plt.tight_layout()

plt.savefig('variants_offline_linear_critical.pdf')
# plt.savefig('variants_offline_linear_critical_eta=%0.2f.pdf' % eta)

plt.show()

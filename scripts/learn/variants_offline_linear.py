"""
Compare different FA variants on offline linear transform problem.
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
n = 3000
# n = 6000
# n = 9000
# n = 12000

epochs = 1

# eta = 0.1
# eta = 5e-2
eta = 2e-2
# eta = 1e-2

# momentum = 0
# momentum = 0.5

alpha = 0
# alpha = 1e-8
# alpha = 1e-6

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

# --- initial weights
# weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
weights = initial_weights(sizes, kind='gaussian', scale=5e-2, rng=rng)
# weights = initial_weights(sizes, kind='gaussian', scale=1e-2, rng=rng)
print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

tau_rc = 0.05
# amp = 0.01
amp = 0.025
# amp = 0.0253

# f, df = static_f_df('lifsoftlif', tau_rc=tau_rc, amplitude=amp)
# f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp, damplitude=2)
f, df = static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp, damplitude=2)

x = np.linspace(-1, 1, 10001)
print("df max: %0.3f" % df(x).max())

genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2)
Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
Bs_direct = [genB((dout, dhid)) for dhid in dhids]

def combine_Bs(Bs):
    Bs_combined = [Bs[-1]]
    for B in Bs[-2::-1]:
        Bs_combined.insert(0, np.dot(Bs_combined[0], B))
    return Bs_combined

print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
print("Bc norms: %s" % ", ".join("%0.3f" % norm(B) for B in combine_Bs(Bs)))
print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

get_network = lambda **kwargs: Network(
    weights, f=f, df=df, biases=None, noise=0, **kwargs)

bp_learner = BPLearner(
    get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
bp_learner.weight_norms = []

# bpl_learner = BPLocalLearner(
#     get_network(), cost, error, eta=eta, alpha=alpha, name='local BP')
# bpl_learner.weight_norms = []

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
# learners = [fa_learner]
# learners = [fas_learner]
# learners = [bp_learner, fa_learner]
learners_u = [bp_learner, fa_learner, fal_learner, fas_learner]
# learners_u = [bp_learner, bpl_learner, fa_learner, fal_learner, fas_learner]

for learner in learners_u:
    learner.train(epochs, batch_fn, test_set=test_set)

for learner in learners_u:
    print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                    for i, w in enumerate(learner.network.weights)))

# --- normalize Bs and learn again
for B, Bc, Bd in zip(Bs, combine_Bs(Bs), Bs_direct):
    B *= norm(B) / norm(Bc)

print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
print("Bc norms: %s" % ", ".join("%0.3f" % norm(B) for B in combine_Bs(Bs)))
print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

fan_learner = FALearner(
    get_network(), cost, error, eta=eta, alpha=alpha, name='global FA')
fan_learner.Bs = [np.array(B) for B in Bs]

faln_learner = FALocalLearner(
    get_network(), cost, error, eta=eta, alpha=alpha, name='local FA')
faln_learner.Bs = [np.array(B) for B in Bs]

learners_n = list(learners_u)
if fa_learner in learners_n:
    learners_n[learners_n.index(fa_learner)] = fan_learner
if fal_learner in learners_n:
    learners_n[learners_n.index(fal_learner)] = faln_learner

for learner in learners_n:
    if len(learner.batch_errors) == 0:
        learner.train(epochs, batch_fn, test_set=test_set)

for learner in learners_n:
    print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                    for i, w in enumerate(learner.network.weights)))


# --- plot results (cols=[unnormalized, normalized], traces=learners)
learner_groups = (learners_u, learners_n)
learner_group_names = ('Unnormalized B', 'Normalized B')
cols = len(learner_groups)
rows = 1

plt.figure(figsize=(6.4, 4))

filt = Alpha(30, default_dt=n_per_batch)

for i, (learners, name) in enumerate(zip(learner_groups, learner_group_names)):
    ax = plt.subplot(rows, cols, i+1)

    for learner in learners:
        x = learner.batch_errors / Yrms
        y = filt.filtfilt(x) if len(x) > 0 else []
        batch_inds = n_per_batch * np.arange(len(x))
        ax.semilogy(batch_inds, y, label=learner.name)

    plt.ylim([1e-1, 1.2e0])
    plt.xlabel('# of examples')
    plt.ylabel('normalized RMS error')
    plt.legend(loc=1)
    plt.title(name)

sns.despine()
plt.tight_layout()

plt.savefig('variants_offline_linear.pdf')

plt.show()

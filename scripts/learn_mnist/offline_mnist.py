"""
Learning MNIST with feedback alignment

- Seems to work all right without momentum
- Orthogonalizing initial weight matrices helps

DTP:
- Works with linear derivatives both forward and back
"""
import matplotlib.pyplot as plt
import numpy as np

from nengo.dists import UniformHypersphere
from nengo.synapses import Alpha

import nengo_extras.matplotlib as nplt
from nengo_extras.data import load_mnist

from hunse_thesis.neurons import static_f_df
from hunse_thesis.offline_learning import (
    Network, ShallowLearner, BPLearner, FALearner, FASkipLearner,
    make_flat_batch_fn)
from hunse_thesis.offline_learning import (
    squared_cost_on_inds, rms_error_on_inds,
    nll_cost_on_inds, class_error_on_inds,
    pointer_squared_cost_on_inds, pointer_class_error_on_inds)

from hunse_thesis.utils import initial_weights, initial_w

rng = np.random.RandomState(9)

# dhids = []
# dhids = [100]
# dhids = [200]
dhids = [500, 500]

# n_per_batch = 1
# n_per_batch = 5
# n_per_batch = 10
# n_per_batch = 20
n_per_batch = 100


(trainX, trainY), (testX, testY) = load_mnist('~/data/mnist.pkl.gz')
labels = np.unique(trainY)
n_labels = len(labels)

# trainX, trainY = trainX[:100], trainY[:100]  # quick training set
# trainX, trainY = trainX[:1000], trainY[:1000]  # quick training set

def preprocess(images):
    images[:] *= 2
    images[:] -= 1

preprocess(trainX), preprocess(testX)

test_subset = testX, testY
# test_subset = test[0][:1000], test[1][:1000]

objective = 'nll'
# objective = 'rms'
# objective = 'sp'
if objective == 'nll':
    cost = nll_cost_on_inds
    error = class_error_on_inds
    dout = n_labels
    dtp_cheat = True
elif objective == 'rms':
    cost = squared_cost_on_inds
    error = rms_error_on_inds
    dout = n_labels
    dtp_cheat = True
elif objective == 'sp':
    dout = 50
    pointers = UniformHypersphere(surface=True).sample(n_labels, d=dout, rng=rng)
    dtp_cheat = False
    def cost(y, yinds, pointers=pointers):
        return pointer_squared_cost_on_inds(y, yinds, pointers)
    def error(y, yinds, pointers=pointers):
        return pointer_class_error_on_inds(y, yinds, pointers)

din = trainX.shape[1]
sizes = [din] + dhids + [dout]

batch_fn = make_flat_batch_fn(trainX, trainY, n_per_batch)

# weights = initial_weights(sizes, kind='uniform', scale=0.001, rng=rng)
weights = initial_weights(sizes, kind='ortho', scale=1, rng=rng)

f, df = static_f_df('liflinear', tau_rc=0.05, amplitude=0.024)

# eta = 1e-1
# eta = 5e-2
# eta = 2e-2
# eta = 1e-2
eta = 4e-3
# eta = 1e-3
# eta = 5e-4
# eta = 2e-4
# eta = 1e-4
# eta = 1e-5
# eta = 0

# alpha = 0
alpha = 1e-6

# --- network
noise = 0
# noise = 1.
get_network = lambda **kwargs: Network(
    weights, f=f, df=df, biases=None, noise=noise, **kwargs)

# --- learners
shallow_learner = ShallowLearner(
    get_network(), cost, error, eta=eta, alpha=alpha)
bp_learner = BPLearner(
    get_network(), cost, error, eta=eta, alpha=alpha)
# bp_learner.weight_norms = []

# fa_learner = FALearner(
#     get_network(), cost, error, eta=eta, alpha=alpha)
# fa_learner.Bs = [initial_w((j, i), kind='ortho', scale=2)
#                  for i, j in zip(dhids, dhids[1:] + [dout])]
# fa_learner.bp_angles = []

fas_learner = FASkipLearner(
    get_network(), cost, error, eta=eta, alpha=alpha)
fas_learner.Bs = [initial_w((dout, dhid), kind='ortho', scale=2) for dhid in dhids]


# epochs = 1
# epochs = 3
epochs = 10
# epochs = 30

# learners = [shallow_learner, bp_learner, fas_learner]
learners = [bp_learner, fas_learner]

for learner in learners:
    learner.train(epochs, batch_fn, test_set=test_subset)

# --- plot
plt.figure(1)
plt.clf()

plt.subplot(211)
def plot_batches(x, label=None):
    filt = Alpha(200, default_dt=n_per_batch)
    y = filt.filtfilt(x) if len(x) > 0 else []
    batch_inds = n_per_batch * np.arange(len(x))
    plt.semilogy(batch_inds, y, label=label)

for learner in learners:
    plot_batches(learner.batch_costs, label=type(learner).__name__)
plt.ylabel('train cost')
plt.legend(loc='best')

plt.subplot(212)
def plot_epochs(x, label=None):
    epoch_inds = len(trainX) * np.arange(1, len(x)+1)
    plt.plot(epoch_inds, x, label=label)

for learner in learners:
    plot_epochs(learner.test_errors, label=type(learner).__name__)
plt.xlabel('example')
plt.ylabel('test error')
plt.legend(loc='best')

# plot encoders
plt.figure(2)
rows, cols = len(learners), 1
for i, learner in enumerate(learners):
    encoders = learner.network.weights[0].T.reshape(-1, 28, 28)
    plt.subplot(rows, cols, i+1)
    nplt.tile(encoders, grid=True)

plt.show()

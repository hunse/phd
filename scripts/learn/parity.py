"""
Parity problem generalized to real domain.

Let XOR of two numbers be 0.5*(|x + y| - |x - y|). Kind of like product but
with absolute values instead of squares.

To recognize each pattern with a different first-layer neuron, we'd need
2**din neurons in the first layer. For anything less than this,
there must be some hierarchical computation going on.

NOTES:
- Seems very prone to getting stuck in local minima. This goes away with
  more neurons.
- moving up to a 32-bit problem has huge effects on generalization. BP is able
  to learn training set, but not generalize to testing.

TODO:
- Try with local-gradient BP
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.utils.numpy import norm

from hunse_thesis.neurons import static_f_df
from hunse_thesis.offline_learning import (
    Network, BPLearner, BPLocalLearner, FASkipLearner, squared_cost, rms_error,
    make_flat_batch_fn)
from hunse_thesis.utils import initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

# rng = np.random
rng = np.random.RandomState(9)

xor = np.frompyfunc(lambda x, y: 0.5*(np.abs(x + y) - np.abs(x - y)), 2, 1)

def xor_reduce(a, **kwargs):
    return xor.reduce(a, **kwargs).astype(a.dtype)

def weight_norm_s(weights):
    return ", ".join("||W%d|| = %0.3f" % (i, norm(w))
                     for i, w in enumerate(weights))

def binary_classification_error(y, ystar):
    return np.sign(y) != ystar


# --- problem
# din = 8
# din = 16
din = 32

# dhids = [16, 8, 4]
# dhids = [24, 12, 6]
# dhids = [32, 16, 8]
# dhids = [20, 20, 20]
# dhids = [50, 50, 50]
# dhids = [64, 64, 64]
# dhids = [64, 48, 32]
dhids = [128, 96, 64]
# dhids = [100, 100, 100]
dout = 1

# ntrain = 1000
# ntest = 100

# ntrain = 10000
# ntest = 3000

ntrain = 50000
ntest = 10000

# epochs = 5
epochs = 50

Xtrain = 2.*rng.randint(0, 2, size=(ntrain, din)) - 1
Ytrain = xor_reduce(Xtrain, axis=1, keepdims=1)

Xtest = 2.*rng.randint(0, 2, size=(ntest, din)) - 1
Ytest = xor_reduce(Xtest, axis=1, keepdims=1)

cost = squared_cost
# error = rms_error
error = binary_classification_error

# --- learn

# weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=0.1)
# weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=0.03, rng=rng)
weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=0.1, rng=rng)
# weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=0.2, rng=rng)
# weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=0.4, rng=rng)
# biases = 0


def decoder(n):
    W = np.zeros((n, n // 4))
    for i in range(n // 4):
        W[4*i:4*(i+1), i] = 0.5 * np.array([1, 1, -1, -1])
    return W

def encoder(n):
    W = np.zeros((n // 2, n))
    for i in range(n // 4):
        W[2*i:2*(i+1), 4*i:4*(i+1)] = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]]).T
    return W

# weights[0][:] = encoder(dhids[0])
# weights[1][:] = np.dot(decoder(dhids[0]), encoder(dhids[1]))
# weights[2][:] = np.dot(decoder(dhids[1]), encoder(dhids[2]))
# weights[3][:] = decoder(dhids[2])

f, df = static_f_df('relu')

# eta = 5e-2
# eta = 2e-2
# eta = 1e-2
# eta = 5e-3
# eta = 2e-3
eta = 1e-3

# momentum = 0
momentum = 0.9

get_network = lambda: Network(weights, f=f, df=df)
# network = Network(weights, biases=biases, f=f, df=df)
bp_learner = BPLearner(get_network(), cost, error,
                       eta=eta, momentum=momentum, name='BP')

bpl_learner = BPLocalLearner(get_network(), cost, error,
                             eta=0.5*eta, momentum=momentum, name='BPLocal')

fas_learner = FASkipLearner(get_network(), cost, error,
                            # eta=eta, momentum=momentum, name='FA')
                            eta=0.5*eta, momentum=momentum, name='FA')
# genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2, rng=rng)
# genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.15, rng=rng)
genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.1, rng=rng)
fas_learner.Bs = [genB((dout, dhid)) for dhid in dhids]

# learners = [bp_learner]
# learners = [bpl_learner]
# learners = [fas_learner]
learners = [bp_learner, bpl_learner, fas_learner]


batch_size = 10
batch_fn = make_flat_batch_fn(Xtrain, Ytrain, batch_size)
for learner in learners:
    print("%s: %s" % (learner.name, weight_norm_s(learner.network.weights)))
    if hasattr(learner, 'Bs'):
        print("%s B: %s" % (learner.name, weight_norm_s(learner.Bs)))

    # learner.train(5, batch_fn, test_set=(Xtest, Ytest))
    # learner.train(10, batch_fn, test_set=(Xtest, Ytest))
    # learner.train(15, batch_fn, test_set=(Xtest, Ytest))
    # learner.train(25, batch_fn, test_set=(Xtest, Ytest))
    # learner.train(50, batch_fn, test_set=(Xtest, Ytest))
    learner.train(epochs, batch_fn, test_set=(Xtest, Ytest))

    # print(learner.test((Xtest, Ytest)).mean())

    # print(np.round(learner.network.weights[0].T, 3))

# --- train/test costs plot
rows = 1
cols = 3

plt.figure()
ax = plt.subplot(rows, cols, 1)
for learner in learners:
    x = learner.train_costs
    epoch_inds = np.arange(len(x))
    ax.semilogy(epoch_inds, x, label=learner.name)
ax.set_xlabel('epochs')
ax.set_ylabel('train cost')

ax = plt.subplot(rows, cols, 2)
for learner in learners:
    x = learner.train_errors
    epoch_inds = np.arange(len(x))
    ax.semilogy(epoch_inds, x, label=learner.name)
ax.set_xlabel('epochs')
ax.set_ylabel('train error')

ax = plt.subplot(rows, cols, 3)
for learner in learners:
    x = learner.test_errors
    epoch_inds = np.arange(len(x))
    ax.semilogy(epoch_inds, x, label=learner.name)
ax.set_xlabel('epochs')
ax.set_ylabel('test error')

sns.despine()
plt.tight_layout()

# --- error distributions plot
rows = 1
cols = len(learners)

plt.figure()
for i, learner in enumerate(learners):
    ax = plt.subplot(rows, cols, i+1)
    errors = learner.test((Xtest, Ytest))
    # ax.hist(errors, bins=19)
    ax.hist(errors, bins=np.linspace(0, 1, 11))
    ax.set_yscale('log')

plt.show()

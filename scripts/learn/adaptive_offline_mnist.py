"""
Try out rules for adapting FA back weights

Notes:
- Symm seems to help even when only applied to earlier layers

IDEAS:
- subtract out activity of random projection (anti-hebbian)
"""
import os

from cycler import cycler
import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.synapses import Alpha
from nengo.utils.numpy import norm
from nengo_extras.data import load_mnist, one_hot_from_labels

from hunse_thesis.neurons import static_f_df
from hunse_thesis.offline_learning import (
    Network, BPLearner, FASkipLearner, make_random_batch_fn)
from hunse_thesis.offline_learning import (
    nll_cost_on_inds, class_error_on_inds)

from hunse_thesis.utils import initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

default_colors = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
plt.rcParams['axes.prop_cycle'] = cycler('color', [(0., 0., 0.)] + default_colors)


class FASymmLearner(FASkipLearner):
    def update(self, t, acts, outs, dC):
        from hunse_thesis.offline_learning import batch_layer_scalar

        etas = batch_layer_scalar(self.eta, t, self.network.n_layers)
        alphas = [0.01 for eta in etas]

        # --- update Bs
        Bs = self.Bs
        dBs = []
        for v in outs[1:-1]:
            dBs.append(-np.dot(dC.T, v))

        vBs = [None] * len(Bs)
        self._update_param(Bs, vBs, dBs, etas, alphas=alphas)


class FAHebbLearner(FASkipLearner):
    def update(self, t, acts, outs, dC):
        from hunse_thesis.offline_learning import batch_layer_scalar

        etas = batch_layer_scalar(self.eta, t, self.network.n_layers)
        etas = [0.1 * eta for eta in etas]
        # etas = [0.5 * eta for eta in etas]

        # alphas = None
        # alphas = [0.001 for eta in etas]
        alphas = [0.01 for eta in etas]
        # alphas = [0.02 for eta in etas]
        # alphas = [0.1 for eta in etas]

        # --- update Bs
        Bs = self.Bs
        dBs = []

        for B in Bs:
            e = np.dot(dC, B)
            dBs.append(np.dot(dC.T, e))

        vBs = [None] * len(Bs)
        self._update_param(Bs, vBs, dBs, etas, alphas=alphas)


class FAOjaLearner(FASkipLearner):
    def update(self, t, acts, outs, dC):
        from hunse_thesis.offline_learning import batch_layer_scalar

        etas = batch_layer_scalar(self.eta, t, self.network.n_layers)
        etas = [0.1 * eta for eta in etas]
        # etas = [0.5 * eta for eta in etas]
        # etas = [1.0 * eta for eta in etas]

        # alphas = None
        # alphas = [0.001 for eta in etas]
        alphas = [0.01 for eta in etas]
        # alphas = [0.1 for eta in etas]

        # beta = 0.1
        # beta = 0.3
        # beta = 0.5
        beta = 1.0
        # beta = 2.0
        # beta = 3.0
        # beta = 10.0

        # deltas = self._deltas[:-1]

        # --- update Bs
        Bs = self.Bs
        dBs = []
        x = dC
        # for y in deltas:
        for B in Bs:
            y = np.dot(dC, B)
            y2 = (y*y).sum(axis=0)
            # y2 = np.abs(y).sum(axis=0)
            # y2 = y.sum(axis=0)
            dBs.append(np.dot(x.T, y) - beta*B*y2[None, :])

            # x2 = (x*x).sum(axis=0)
            # x2 = np.abs(x).sum(axis=0)
            # x2 = x.sum(axis=0)
            # dBs.append(np.dot(x.T, y) - beta*B*x2[:, None])

        vBs = [None] * len(Bs)
        self._update_param(Bs, vBs, dBs, etas, alphas=alphas)


class FASymmHebbLearner(FASkipLearner):
    def update(self, t, acts, outs, dC):
        from hunse_thesis.offline_learning import batch_layer_scalar

        etas = batch_layer_scalar(self.eta, t, self.network.n_layers)
        alphas = [0.01 for eta in etas]
        upsilon = 0.1  # scaling on unsupervised

        # --- update Bs
        Bs = self.Bs
        dBs = []

        dBs = []
        for v, B in zip(outs[1:-1], Bs):
            e = v - upsilon*np.dot(dC, B)
            dBs.append(-np.dot(dC.T, e))

        vBs = [None] * len(Bs)
        self._update_param(Bs, vBs, dBs, etas, alphas=alphas)


class FASymmOjaLearner(FASkipLearner):
    def update(self, t, acts, outs, dC):
        from hunse_thesis.offline_learning import batch_layer_scalar

        etas = batch_layer_scalar(self.eta, t, self.network.n_layers)
        alphas = [0.01 for eta in etas]
        beta = 1.0
        # upsilon = 0.5  # scaling on unsupervised
        upsilon = 0.1  # scaling on unsupervised

        # --- update Bs
        Bs = self.Bs
        dBs = []
        for v, B in zip(outs[1:-1], Bs):
            y = np.dot(dC, B)
            y2 = (y*y).sum(axis=0)
            # dBsymm = -np.dot(dC.T, v)
            # dBoja = np.dot(dC.T, y) - beta*B*y2[None, :]
            # dBs.append(dBsymm + upsilon*dBoja)
            dB0 = np.dot(dC.T, -v + upsilon*y)
            dB1 = -((upsilon*beta)*y2[None, :])*B
            dBs.append(dB0 + dB1)

        vBs = [None] * len(Bs)
        self._update_param(Bs, vBs, dBs, etas, alphas=alphas)


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

if 0:
    # rotational task
    rng = np.random.RandomState(5)
    def rotate(X):
        X = X.reshape(-1, 28, 28)
        rots = rng.randint(0, 4, size=X.shape[0])
        flip = rng.randint(0, 2, size=X.shape[0]).astype(bool)
        for i in range(4):
            m = rots == i
            X[m] = np.rot90(X[m], k=i, axes=(1, 2))
        X[flip] = X[flip][:, :, ::-1]

    rotate(trainX)
    rotate(testX)

# --- experiment
# filename = None
# filename = 'adaptive_offline_mnist.dil'
# filename = 'adaptive_offline_mnist_test.dil'
# filename = 'adaptive_offline_mnist_scale=0.5.dil'
filename = 'adaptive_offline_mnist_scale=0.5_minoja.dil'

if filename is None or not os.path.exists(filename):
    seed = 1
    # seed = np.random.randint(1000000)

    rng = np.random.RandomState(seed)
    print("Seed: %d" % seed)

    # --- params
    # dhids = [200, 200]
    dhids = [500, 500]

    # eta = 0.1
    # eta = 5e-2
    # eta = 2e-2
    # eta = 1e-2
    # eta = 5e-3
    eta = 2e-3
    # eta = 1e-3

    alpha = 0

    # epochs = 1
    # epochs = 2
    # epochs = 3
    # epochs = 10
    # epochs = 15
    epochs = 25

    # n_per_batch = 1
    # n_per_batch = 2
    # n_per_batch = 5
    # n_per_batch = 10
    n_per_batch = 20
    # n_per_batch = 100

    # batch_fn = make_flat_batch_fn(trainX, trainY, n_per_batch)
    batch_fn = make_random_batch_fn(trainX, trainY, n_per_batch,
                                    rng=np.random.RandomState(5))

    sizes = [din] + dhids + [dout]

    # --- initial weights
    weights = initial_weights(sizes, kind='gaussian', scale=1e-1, rng=rng)
    print(", ".join("||W%d|| = %0.3f" % (i, norm(w)) for i, w in enumerate(weights)))

    # genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.2)
    # genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.3)
    # genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.4)
    genB = lambda shape: initial_w(shape, rng=rng, kind='ortho', normkind='rightmean', scale=0.5)
    Bs_direct = [genB((dout, dhid)) for dhid in dhids]

    # --- nonlinearity
    tau_rc = 0.05
    # amp = 0.01
    amp = 0.025
    f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp)

    # --- learners
    get_network = lambda **kwargs: Network(
        weights, f=f, df=df, biases=None, noise=0, **kwargs)

    bp_learner = BPLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
    # bp_learner.weight_norms = []
    bp_learner.delta_norms = []

    fas_learner = FASkipLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='DFA')
    fas_learner.Bs = [B.copy() for B in Bs_direct]
    fas_learner.delta_norms = []
    fas_learner.bpd_angles = []

    fay_learner = FASymmLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='Symmetric ADFA')
    fay_learner.Bs = [B.copy() for B in Bs_direct]
    fay_learner.delta_norms = []
    fay_learner.bpd_angles = []

    fah_learner = FAHebbLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='Hebbian ADFA')
    fah_learner.Bs = [B.copy() for B in Bs_direct]
    fah_learner.delta_norms = []
    fah_learner.bpd_angles = []

    fao_learner = FAOjaLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='Oja ADFA')
    fao_learner.Bs = [B.copy() for B in Bs_direct]
    fao_learner.delta_norms = []
    fao_learner.bpd_angles = []

    fayh_learner = FASymmHebbLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='Symm-Hebb ADFA')
    fayh_learner.Bs = [B.copy() for B in Bs_direct]
    fayh_learner.delta_norms = []
    fayh_learner.bpd_angles = []

    fayo_learner = FASymmOjaLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='Symm-Oja ADFA')
    fayo_learner.Bs = [B.copy() for B in Bs_direct]
    fayo_learner.delta_norms = []
    fayo_learner.bpd_angles = []

    # learners = [bp_learner]
    # learners = [fas_learner]
    # learners = [bp_learner, fas_learner]
    # learners = [bp_learner, fas_learner, fay_learner]
    # learners = [fay_learner]
    # learners = [fas_learner, fay_learner]
    # learners = [fah_learner]
    # learners = [fas_learner, fah_learner]
    # learners = [fayh_learner]
    # learners = [fay_learner, fayh_learner]
    # learners = [fas_learner, fayh_learner]
    # learners = [fas_learner, fay_learner, fayh_learner]
    # learners = [fas_learner, fay_learner, fah_learner, fayh_learner]
    # learners = [fao_learner]
    # learners = [fah_learner, fao_learner]
    learners = [bp_learner, fas_learner, fay_learner, fah_learner, fao_learner, fayh_learner, fayo_learner]
    for learner in learners:
        learner.train(epochs, batch_fn, test_set=test_set)

    if filename is not None:
        with open(filename, 'wb') as fh:
            dill.dump(dict(
                seed=seed, dhids=dhids, epochs=epochs, n_per_batch=n_per_batch,
                eta=eta, alpha=alpha, weights=weights, tau_rc=tau_rc, amp=amp,
                learners=learners), fh)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)

# hide Hebbian learner (unstable)
learners = [learner for learner in learners if 'Hebbian' not in learner.name]

# --- plot results (cols=[train, test], traces=learners)
fig = plt.figure(figsize=(6.4, 5))
rows = 2
cols = 2

n_batches = len(learners[0].batch_errors)
batch_inds = (n_per_batch / 1000.) * np.arange(n_batches)
epoch_inds = (trainX.shape[0] / 1000.) * np.arange(1, epochs+1)

# - train subplot
ax = fig.add_subplot(rows, cols, 1)

# filt = Alpha(1000, default_dt=n_per_batch)
# filt = Alpha(3000, default_dt=n_per_batch)
filt = Alpha(10000, default_dt=n_per_batch)

for learner in learners:
    y = filt.filtfilt(learner.batch_errors)
    ax.semilogy(batch_inds, y, label=learner.name)

# plt.ylim([1e-4, 5e-1])
plt.ylim([None, 2e-1])
plt.xlabel('thousands of examples')
plt.ylabel('train error')
plt.legend(loc='best')
# plt.title("Train error")

# - test subplot
ax = fig.add_subplot(rows, cols, 2)

filt = Alpha(0.5, default_dt=1)
# filt = Alpha(1, default_dt=1)

for learner in learners:
    y = filt.filtfilt(learner.test_errors)
    plt.semilogy(epoch_inds, y, label=learner.name)

plt.ylim([1e-2, 1e-1])
plt.xlabel('thousands of examples')
plt.ylabel('test error')
# plt.legend(loc=1)
# plt.title("Test error")

# - delta norms
ax = fig.add_subplot(rows, cols, 3)

filt = Alpha(10000, default_dt=n_per_batch)
delta_norms = np.array([learner.delta_norms for learner in learners])
delta_norms = filt.filtfilt(delta_norms, axis=1)
plt.semilogy(batch_inds, delta_norms[:, :, 0].T, '-')
ax.set_color_cycle(None)
plt.semilogy(batch_inds, delta_norms[:, :, 1].T, '--')
# plt.plot(batch_inds, delta_norms[:, :, 0].T, '-')
# plt.plot(batch_inds, delta_norms[:, :, 1].T, '--')

plt.ylim([None, 1])
plt.xlabel('thousands of examples')
plt.ylabel(r'$\delta$ norm')
plt.legend(loc=1)

# - angles
ax = fig.add_subplot(rows, cols, 4)

filt = Alpha(10000, default_dt=n_per_batch)
bpd_angles = np.array([
    getattr(learner, 'bpd_angles', np.nan*np.ones((n_batches, len(dhids))))
    for learner in learners])
bpd_angles = filt.filtfilt(bpd_angles, axis=1)
plt.plot(batch_inds, bpd_angles[:, :, 0].T * (180/np.pi), '-')
ax.set_color_cycle(None)
plt.plot(batch_inds, bpd_angles[:, :, 1].T * (180/np.pi), '--')

plt.ylim([0, 90])
plt.xlabel('thousands of examples')
plt.ylabel(r'angle [degrees]')
plt.legend(loc=1)

sns.despine()
plt.tight_layout()
plt.savefig('adaptive_offline_mnist.pdf')

plt.show()

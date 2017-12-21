"""
Examine selectivity of BP and FA neurons on MNIST
"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms
from nengo_extras.convnet import softmax
from nengo_extras.data import load_mnist, one_hot_from_labels

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, BPLearner, FALearner, FALocalLearner, FASkipLearner,
    make_flat_batch_fn, make_random_batch_fn)
from hunse_thesis.offline_learning import (
    nll_cost_on_inds, class_error_on_inds)

from hunse_thesis.utils import orthogonalize, initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

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
# filename = 'selectivity_offline_mnist_all2.dil'
filename = 'selectivity_offline_mnist_eye2.dil'

# filename = 'selectivity_offline_mnist_all.dil'
# filename = 'selectivity_offline_mnist_rot.dil'
# filename = 'selectivity_offline_mnist_identity.dil'
# filename = 'selectivity_offline_mnist_identityinhib.dil'

if not os.path.exists(filename):
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

    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2)
    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.4)
    # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.1)
    genB = lambda shape: initial_w(shape, kind='identity', normkind='rightmean', scale=0.2)
    Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
    Bs_direct = [genB((dout, dhid)) for dhid in dhids]

    def combine_Bs(Bs):
        Bs_combined = [Bs[-1]]
        for B in Bs[-2::-1]:
            Bs_combined.insert(0, np.dot(Bs_combined[0], B))
        return Bs_combined

    for B, Bc, Bd in zip(Bs, combine_Bs(Bs), Bs_direct):
        B *= norm(B) / norm(Bc)

    print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
    print("Bc norms: %s" % ", ".join("%0.3f" % norm(B) for B in combine_Bs(Bs)))
    print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

    # --- nonlinearity
    tau_rc = 0.05
    # amp = 0.01
    amp = 0.025
    # amp = 0.0253

    f, df = static_f_df('liflinear', tau_rc=tau_rc, amplitude=amp)
    # f, df = static_f_df('lifstep', tau_rc=tau_rc, amplitude=amp)

    # --- learners
    get_network = lambda **kwargs: Network(
        weights, f=f, df=df, biases=None, noise=0, **kwargs)

    bp_learner = BPLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, name='BP')
    bp_learner.weight_norms = []

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
    # learners = [fas_learner]
    # learners = [bp_learner, fas_learner]
    # learners = [bp_learner, fa_learner]
    learners = [bp_learner, fa_learner, fal_learner, fas_learner]

    for learner in learners:
        learner.train(epochs, batch_fn, test_set=test_set)

    for learner in learners:
        print(", ".join("||W%d|| = %0.3f" % (i, norm(w))
                        for i, w in enumerate(learner.network.weights)))

    with open(filename, 'wb') as fh:
        dill.dump(dict(
            seed=seed, dhids=dhids, epochs=epochs, n_per_batch=n_per_batch,
            eta=eta, alpha=alpha, weights=weights, tau_rc=tau_rc, amp=amp,
            learners=learners), fh)
    print("Saved %r" % filename)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)

# --- sanity test (to check that everything saved fine)
# for learner in learners:
#     errors = learner.test(test_set)
#     print('%s: %0.3f' % (learner.name, errors.mean()))

# --- plot weighted selectivity (rows=learners, cols=layers, traces=neurons)
testT = one_hot_from_labels(testY, classes=10).astype(float)
testW = testT / testT.sum(0)
test_outs = [learner.network.forward(testX)[1] for learner in learners]

# fig_sel = plt.figure(figsize=(6.4, 3.5))
# rows = len(learners)
# cols = learners[0].network.n_layers

# for i, outs in enumerate(test_outs):
#     # outs = outs[1:-1]
#     outs = outs[1:]
#     outs[-1][:] = softmax(outs[-1], axis=1)

#     for j, h in enumerate(outs):
#         h = h / (h.sum(0) + 1e-16)
#         R = np.dot(h.T, testT)  # responses: neurons x classes

#         # plot selectivities for each neuron
#         ax = fig_sel.add_subplot(rows, cols, i*cols+j+1)
#         # ax.plot(np.arange(10), R[:300].T)
#         # ax.plot(np.arange(10), R.T, '.')

#         k = np.argmax(R, axis=1)
#         ax.plot(k, R[np.arange(R.shape[0]), k], '.')
#         ax.set_xlim([-0.5, 9.5])
#         ax.set_ylim([0, 1.1])
#         if i == 0:
#             ax.set_title('layer %d' % (j+1))

# sns.despine(fig_sel)
# fig_sel.tight_layout()

# --- plot selectivity histogram
fig_selhist = plt.figure(figsize=(6.4, 3.5))
# rows = 1
rows = 2
cols = learners[0].network.n_layers - 1

bin_edges = np.linspace(-0.5, 9.5, 11)
bin_mids = 0.5*(bin_edges[1:] + bin_edges[:-1])

axes = [fig_selhist.add_subplot(rows, cols, j+1) for j in range(cols)]
axes2 = [fig_selhist.add_subplot(rows, cols, cols+j+1) for j in range(cols)]

for i, outs in enumerate(test_outs):
    outs = outs[1:-1]

    for j, h in enumerate(outs):
        hc_means = h.T.dot(testW)
        hc_stds = np.sqrt((h**2).T.dot(testW) - hc_means**2)
        hc_stes = hc_stds / np.sqrt(testT.sum(0))[None, :]
        h_mean = h.mean(0)

        hc_z = (hc_means - h_mean[:, None]) / (hc_stds + 1e-16)
        hc_q = (hc_means - h_mean[:, None]) / (hc_stes + 1e-16)
        print(hc_z[0])
        print(hc_q[0])

        # threshold = 1.96
        # # Rcount = (hc_means - h_mean[:, None] > threshold * hc_stes).sum(axis=1)
        # Rcount = (hc_q > threshold).sum(axis=1)

        # threshold = 1.0
        threshold = 0.5
        # Rcount = (hc_z > threshold).sum(axis=1)
        Rcount = (np.abs(hc_z) > threshold).sum(axis=1)

        counts, _ = np.histogram(Rcount, bins=bin_edges)
        axes[j].plot(bin_mids, (100./counts.sum())*counts, 'o--', label=learners[i].name)


# Ss = [[None for _ in outs[1:-1]] for outs in test_outs]
# for i, outs in enumerate(test_outs):
#     outs = outs[1:-1]

#     for j, h in enumerate(outs):
#         # print("%s %d: %s" % (learners[i].name, j, (h.max(axis=0) < 1e-10).mean()))
#         hz = (h - h.mean(0)) / (h.std(0) + 1e-16)
#         R = np.dot(hz.T, testW)

#         thresholds = 1.96 / np.sqrt(testT.sum(0))
#         Rcount = (R > thresholds).sum(axis=1)

#         # plot number of selective neurons
#         counts, _ = np.histogram(Rcount, bins=bin_edges)
#         axes[j].plot(bin_mids, (100./counts.sum())*counts, 'o--', label=learners[i].name)

#         # plot histogram of selectivity statistic
#         S = np.zeros(hz.shape[1])
#         binsx = int(np.sqrt(testW.shape[0]))
#         # print(binsx)
#         # binsx = 31
#         # binsx = 51
#         for k, x in enumerate(hz.T):
#             # add noise to ensure no identical values
#             nradius = 1e-10
#             x = x + np.random.uniform(-nradius, nradius, size=x.shape)

#             # partition into equal count bins
#             xs = np.sort(x)
#             blocks = np.array_split(xs, binsx)
#             edges = [xs[0]] + [0.5*(b0[-1] + b1[0]) for b0, b1 in zip(blocks[:-1], blocks[1:])] + [xs[-1]]

#             # edges = np.linspace(x.min(), x.max(), binsx)

#             # histogram
#             pXY = np.histogram2d(x, testY, bins=(edges, 10))[0]
#             pXY /= pXY.sum()
#             pX = pXY.sum(1)
#             pY = pXY.sum(0)
#             assert np.isfinite(pXY).all()

#             # compute mutual information and Y entropy
#             M = pXY * np.log(pXY / (pX[:, None] * pY[None, :]))
#             M[pXY == 0] = 0
#             assert np.isfinite(M).all()
#             EY = -pY * np.log(pY)
#             EY[pY == 0] = 0
#             S[k] = M.sum() / EY.sum()

#         Ss[i][j] = S

# maxS = max(S.max() for Ssi in Ss for S in Ssi)
# bins = np.linspace(0, maxS, 16)
# binsc = 0.5*(bins[:-1] + bins[1:])
# for i, Ssi in enumerate(Ss):
#     for j, S in enumerate(Ssi):
#         counts, _ = np.histogram(S, bins=bins)
#         axes2[j].plot(binsc, (100./counts.sum())*counts, 'o--', label=learners[i].name)

for j, ax in enumerate(axes):
    ax.set_xlim([-0.5, 9.5])
    ax.set_ylim([0, 40])
    ax.set_xlabel('# of selective classes')
    if j == 0:
        ax.set_ylabel('percent of neurons')
        # ax.set_ylabel('# of neurons')
    ax.set_title('layer %d' % (j+1))
    ax.legend(loc='best')

for j, ax in enumerate(axes2):
    ax.set_ylim([0, 30])
    ax.set_xlabel('selectivity')
    if j == 0:
        ax.set_ylabel('percent of neurons')
    # ax.legend(loc='best')

sns.despine(fig_selhist)
fig_selhist.tight_layout()
fig_selhist.savefig('selectivity_offline_mnist_selhist.pdf')

# --- plot some hidden neurons and their B vectors
f = learners[0].network.f
df = learners[0].network.df
network0 = Network(weights, f=f, df=df, biases=None, noise=0)
_, outs0 = network0.forward(testX)
outs0 = outs0[1:-1]

learner = learners[-1]  # FASkip learner
_, outs = learner.network.forward(testX)
outs = outs[1:-1]

# plt.figure(figsize=(6.4, 3.5))
# rows = 1
# cols = len(outs)

# for j, h in enumerate(outs):
#     ax = plt.subplot(rows, cols, j+1)

#     h = h / (h.sum(0) + 1e-16)
#     R = np.dot(h.T, testT)  # responses: neurons x classes
#     # ax.plot(np.arange(10), R[:300].T)
#     # ax.plot(np.arange(10), R.T, '.')

#     # k = np.argmax(R, axis=1)
#     # ax.plot(k, R[np.arange(R.shape[0]), k], '.')

#     B = learner.Bs[j]
#     B = (B - B.min(0))
#     B = B / B.sum(0)

#     pair_max = np.maximum(B.max(0), R.max(1))

#     n = 10
#     offsets = 1.5*np.arange(n)
#     ax.plot(np.arange(10), R[:n].T/pair_max[:n] + offsets)
#     ax.set_color_cycle(None)
#     ax.plot(np.arange(10), B[:, :n]/pair_max[:n] + offsets, '--')

#     # for k in range(5):
#     #     ax.plot(np.arange(10), R[k])

#     # ax.set_color_cycle(None)

#     # for k in range(5):
#     #     # ax.plot(np.arange(10), learner.Bs[j][:, k], '--')
#     #     ax.plot(np.arange(10), B[:, k], '--')

#     ax.set_xlim([-0.5, 9.5])
#     # ax.set_ylim([0, 1.1])

# sns.despine()
# plt.tight_layout()

# plot histogram of distances between B vectors and responses
# plt.figure(figsize=(6.4, 3.5))
plt.figure(figsize=(6, 3))
rows = 1
cols = len(outs)

axes = [plt.subplot(rows, cols, j+1) for j in range(cols)]

normalize = lambda X, axis=None: (X - X.mean(axis=axis, keepdims=1)) / (
    X.std(axis=axis, keepdims=1) + 1e-16)

for j, h in enumerate(outs):
    ax = axes[j]
    ax.set_title('layer %d' % (j+1))

    B = learner.Bs[j].T.copy()  # neurons x classes
    B = normalize(B, axis=1)

    # reference (before learning)
    h0 = outs0[j]
    h0 = h0 / (h0.sum(0) + 1e-16)
    R0 = np.dot(h0.T, testT)
    R0 = normalize(R0, axis=1)
    dots0 = (B * R0).mean(1)

    # actual (after learning)
    h = h / (h.sum(0) + 1e-16)
    R = np.dot(h.T, testT)  # responses: neurons x classes
    # R /= norm(R, axis=1, keepdims=1) + 1e-16
    R = normalize(R, axis=1)
    dots = (B * R).mean(1)

    # Rrank = normalize(np.argsort(R, axis=1), axis=1)
    # Brank = normalize(np.argsort(B, axis=1), axis=1)
    # dots_rank = (Brank * Rrank).mean(1)

    # --- plot histograms
    bins = np.linspace(-1, 1, 20)
    binc = 0.5*(bins[:-1] + bins[1:])
    weights = (100./dots.size) * np.ones(dots.shape)

    hist0, _ = np.histogram(dots0, bins=bins, weights=weights)
    ax.plot(binc, hist0, 'k--o')

    hist1, _ = np.histogram(dots, bins=bins, weights=weights)
    ax.plot(binc, hist1, '-o')

    # ax.set_ylim([0, ])
    ax.set_xlabel('correlation coefficient')
    if j == 0:
        ax.set_ylabel('percent of neurons')

for ax in axes:
    ax.set_ylim([0, max(ax.get_ylim()[1] for ax in axes)])

sns.despine()
plt.tight_layout()
plt.savefig('selectivity_offline_mnist_corrhist.pdf')

plt.show()

"""
Classification problem with 2-D input, dout classes that appear in n "bands".

Results:
- With masking on input weights, BP can learn because first layer can
  separate into bands vertically and horizontally, second layer uses this
  to classify. FA can't do this, because it needs to push first-layer neurons
  towards particular categories, and there is no category information
  available in a single input dimension.
- FA problems may have something to do with difficulty of learning biases

Discussion:
- FA problems could be fixed by having many more neurons in second layer,
  such that first layer just needs to encode input, and second layer can
  group the whole 2D space into n^2 regions. However, this is avoiding the
  real need to learn compositional functions, and will not scale to difficult
  problems which really require deep networks.
- Pertinent to deep classification networks, since these typically have local
  (convolutional) early layers.
"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nengo.utils.numpy import norm

from hunse_thesis.neurons import static_f_df
from hunse_thesis.offline_learning import (
    Network, BPLearner, BPLocalLearner, FALearner, FASkipLearner,
    BPLocalLearner2,
    squared_cost, rms_error, make_flat_batch_fn)
from hunse_thesis.offline_learning import (nll_cost_on_inds, class_error_on_inds)
from hunse_thesis.utils import initial_weights, initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')
# sns.set(context='paper', style='ticks', palette='deep')

def weight_norm_s(weights):
    return ", ".join("||W%d|| = %0.3f" % (i, norm(w))
                     for i, w in enumerate(weights))

def pad_ones(X):
    return np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

def band_widths_edges(nbands, margin):
    bandw = (2. - (nbands-1)*margin) / nbands
    band_edges = (margin + bandw)*np.arange(nbands+1) - 1 - 0.5*margin
    return bandw, band_edges


# CONSTANT_INPUT = False
CONSTANT_INPUT = True

# mask_states = [False]
mask_states = [False, True]

filename = 'banded2d_9band.dil'

if not os.path.exists(filename):
    # seed = np.random.randint(2**10)
    seed = 1

    print("Seed: %d" % seed)
    rng = np.random.RandomState(seed)

    # --- make data
    # nbands = 2
    # nbands = 3
    # nbands = 4
    # nbands = 6
    # nbands = 8
    nbands = 9

    ntrain = 50000
    # ntest = 10000
    ntest = 20000
    # ntest = 30000

    din = 2

    # dout = 2
    dout = 3

    # -- nonadjacent bands
    margin = 0.3 / nbands

    bandw, band_edges = band_widths_edges(nbands, margin)

    # def class_ij(i, j):
    #     assert din == 2 and dout == 3
    #     # j2 = (j // 2) % 2
    #     # k = (2*j2 - 1)*i + j + j2
    #     k = i + (j % 2)
    #     return k % dout


    def gendata(n):
        bandi = rng.randint(nbands, size=(n, din))
        X = (margin + bandw) * bandi - 1 + rng.uniform(0, bandw, size=(n, din))
        Y = (bandi.sum(axis=1) % dout)
        # Y = class_ij(bandi[:, 0], bandi[:, 1])
        if CONSTANT_INPUT:
            X = pad_ones(X)
        return X, Y

    Xtrain, Ytrain = gendata(ntrain)
    Xtest, Ytest = gendata(ntest)

    cost = nll_cost_on_inds
    error = class_error_on_inds

    all_learners = []
    for use_mask in mask_states:
        # --- learn
        # dhids = [nbands**2, nbands**2]
        # dhids = [2*nbands, 2*nbands]
        # dhids = [3*nbands, 3*nbands]
        dhids = [5*nbands, 5*nbands]

        weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=1., rng=rng)
        # weights = initial_weights([din] + dhids + [dout], kind='uniform', scale=0.5, rng=rng)
        # weights = initial_weights([din] + dhids + [dout], kind='binary', scale=1., rng=rng)

        if use_mask:
            mask_options = np.array([[1, 0], [0, 1]], dtype=bool)
        else:
            mask_options = np.array([[1, 1]], dtype=bool)

        weight_masks = [np.column_stack(mask_options[
            rng.randint(len(mask_options), size=dhids[0])])]

        for W, Wm in zip(weights, weight_masks):
            W *= Wm

        # biases = [np.zeros(d) for d in dhids + [dout]]
        # biases = [-0.5 * np.ones(d) for d in dhids + [dout]]
        biases = [rng.uniform(-1, 1, size=d) for d in dhids + [dout]]
        # biases = [rng.uniform(-0.1, 0.1, size=d) for d in dhids + [dout]]
        # biases = [rng.choice(-band_edges, size=d) for d in dhids + [dout]]
        # biases[-1][:] = 0

        # intercepts = [0.5 * np.ones(d) for d in dhids + [dout]]
        # intercepts = [rng.choice(band_edges, size=d) for d in dhids + [dout]]

        # biases = [-norm(w, axis=0)*t for w, t in zip(weights, intercepts)]
        # biases[-1][:] = 0

        # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=1., rng=rng)
        # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.5, rng=rng)
        # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.3, rng=rng)
        genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.2, rng=rng)
        # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.15, rng=rng)
        # genB = lambda shape: initial_w(shape, kind='ortho', normkind='rightmean', scale=0.1, rng=rng)

        Bs = [genB((d1, d0)) for d0, d1 in zip(dhids, dhids[1:] + [dout])]
        Bs_direct = [genB((dout, dhid)) for dhid in dhids]

        def combine_Bs(Bs):
            Bs_combined = [Bs[-1]]
            for B in Bs[-2::-1]:
                Bs_combined.insert(0, np.dot(Bs_combined[0], B))
            return Bs_combined

        for B, Bc, Bd in zip(Bs, combine_Bs(Bs), Bs_direct):
            B *= norm(B) / norm(Bc)

        # print("B norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs))
        # print("Bc norms: %s" % ", ".join("%0.3f" % norm(B) for B in combine_Bs(Bs)))
        # print("Bd norms: %s" % ", ".join("%0.3f" % norm(B) for B in Bs_direct))

        # --- nonlinearity
        f, df = static_f_df('relu')

        # epochs = 2
        # epochs = 10
        # epochs = 15
        # epochs = 50
        epochs = 100

        # eta = 1e-1
        # eta = 5e-2
        # eta = 2e-2
        # eta = 1e-2
        # eta = 5e-3
        # eta = 2e-3
        # eta = 1e-3
        # eta = 5e-4
        # eta = 2e-4
        eta = 1e-4

        # momentum = 0
        # momentum = 0.5
        momentum = 0.9

        if CONSTANT_INPUT:
            weights[0] = np.vstack((weights[0], biases[0]))
            weight_masks[0] = np.vstack((weight_masks[0], np.ones(dhids[0], dtype=bool)))

            get_network = lambda: Network(weights, f=f, df=df, weight_masks=weight_masks)
        else:
            # get_network = lambda: Network(weights, f=f, df=df)
            # get_network = lambda: Network(weights, biases=biases, f=f, df=df)
            get_network = lambda: Network(weights, biases=biases, f=f, df=df, weight_masks=weight_masks)

        network0 = get_network()

        bp_learner = BPLearner(get_network(), cost, error,
                               eta=eta, momentum=momentum, name='BP')

        bpl_learner = BPLocalLearner(get_network(), cost, error,
                                     # eta=eta, momentum=momentum, name='local BP')
                                     eta=0.1*eta, momentum=momentum, name='local BP')
                                     # eta=0.05*eta, momentum=momentum, name='local BP')

        # bp2_learner = BPLocalLearner2(get_network(), cost, error, name='local BP 2',
        #                               # eta=eta, momentum=momentum)
        #                               # eta=0.1*eta, momentum=momentum)
        #                               eta=0.01*eta, momentum=momentum)

        fa_learner = FALearner(get_network(), cost, error,
                               eta=eta, momentum=momentum, name='GFA')
                               # eta=0.5*eta, momentum=momentum, name='GFA')
        fa_learner.Bs = Bs

        fas_learner = FASkipLearner(get_network(), cost, error,
                                    eta=eta, momentum=momentum, name='DFA')
                                    # eta=0.5*eta, momentum=momentum, name='DFA')
                                    # eta=0.25*eta, momentum=momentum, name='DFA')
                                    # eta=0.1*eta, momentum=momentum, name='DFA')
        fas_learner.Bs = Bs_direct

        learners = [bp_learner, bpl_learner, fa_learner, fas_learner]

        # batch_size = 10
        batch_size = 100
        batch_fn = make_flat_batch_fn(Xtrain, Ytrain, batch_size)
        for learner in learners:
            print("%s: %s" % (learner.name, weight_norm_s(learner.network.weights)))
            if hasattr(learner, 'Bs'):
                print("%s B: %s" % (learner.name, weight_norm_s(learner.Bs)))

            learner.train(epochs, batch_fn, test_set=(Xtest, Ytest))

        all_learners.append(learners)

    # if filename is not None:
    with open(filename, 'wb') as fh:
        dill.dump(dict(
            seed=seed, Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest,
            din=din, dhids=dhids, dout=dout, nbands=nbands, margin=margin,
            network0=network0, all_learners=all_learners), fh)

else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)

    bandw, band_edges = band_widths_edges(nbands, margin)


# --- Dataset plot
plt.figure(figsize=(5, 3.5))
X, Y = Xtrain[:6000], Ytrain[:6000]
for c in range(dout):
    plt.plot(X[Y == c, 0], X[Y == c, 1], '.')
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.tight_layout()
plt.savefig('banded2d_dataset.pdf')

# --- train/test costs plot
# plt.figure()
# plt.figure(figsize=(7, 3.5))
plt.figure(figsize=(6.35, 4))

rows = 1
cols = 2

# ax = plt.subplot(rows, cols, 1)
# for use_mask, learners in zip(mask_states, all_learners):
#     ls = '-' if use_mask else '--'
#     for learner in learners:
#         x = learner.train_costs
#         epoch_inds = np.arange(len(x))
#         ax.semilogy(epoch_inds, x, label=learner.name)
#         ax.set_ylim([None, 2e0])
# ax.set_xlabel('epochs')
# ax.set_ylabel('train cost')
# ax.legend(loc='best')

ax = plt.subplot(rows, cols, 1)
for use_mask, learners in zip(mask_states, all_learners):
    ax.set_color_cycle(None)
    ls = '-' if not use_mask else '--'
    for learner in learners:
        x = learner.train_errors
        epoch_inds = np.arange(len(x))
        ax.plot(epoch_inds, x, ls=ls,
                label=learner.name if not use_mask else None)
ax.set_yscale('log')
ax.set_ylim([3e-2, 1e0])
ax.set_xlabel('epochs')
ax.set_ylabel('train error')
ax.legend(loc='best')

ax = plt.subplot(rows, cols, 2)
for use_mask, learners in zip(mask_states, all_learners):
    ax.set_color_cycle(None)
    ls = '-' if not use_mask else '--'
    for learner in learners:
        x = learner.test_errors
        epoch_inds = np.arange(len(x))
        ax.plot(epoch_inds, x, ls=ls)
ax.set_yscale('log')
ax.set_ylim([3e-2, 1e0])
ax.set_xlabel('epochs')
ax.set_ylabel('test error')

sns.despine()
plt.tight_layout()
plt.savefig('banded2d_losses.pdf')

# --- train-only plot (for presentation)
# plt.figure(figsize=(5, 4))
plt.figure(figsize=(4, 3))
rows = 1
cols = 1

ax = plt.subplot(rows, cols, 1)
for use_mask, learners in zip(mask_states, all_learners):
    ax.set_color_cycle(None)
    ls = '-' if not use_mask else '--'
    for learner in learners:
        x = learner.train_errors
        epoch_inds = np.arange(len(x))
        ax.plot(epoch_inds, x, ls=ls, label=learner.name if use_mask else None)
# ax.set_yscale('log')
# ax.set_ylim([3e-2, 1e0])
ax.set_ylim([0.05, 0.75])
ax.set_xlabel('epochs')
ax.set_ylabel('train error')
ax.legend(loc='best')

sns.despine()
plt.tight_layout()
plt.savefig('banded2d_train.pdf')

# --- visualize what first layer is doing
# def visualize1(network, dim):
#     x = np.linspace(-1, 1, 1001)
#     X = np.zeros((len(x), din))
#     X[:, dim] = x
#     if CONSTANT_INPUT:
#         X = pad_ones(X)

#     y = network.forward1(0, X)[1]
#     if len(network.weight_masks) > 0:
#         y = y[:, network.weight_masks[0][dim, :] > 0]

#     plt.plot(x, y)

#     Xref = Xtrain[:1000, dim]
#     plt.plot(Xref, -0.1 * np.ones_like(Xref), 'k.')
#     plt.plot(band_edges, -0.2 * np.ones_like(band_edges), 'r.')


# plt.figure()

# rows = 1 + len(learners)
# cols = 2

# plt.subplot(rows, cols, 1)
# visualize1(network0, 0)
# plt.title('dim 0')
# plt.subplot(rows, cols, 2)
# visualize1(network0, 1)
# plt.title('dim 1')

# for i, learner in enumerate(learners):
#     plt.subplot(rows, cols, (i+1)*cols + 1)
#     visualize1(learner.network, 0)
#     plt.subplot(rows, cols, (i+1)*cols + 2)
#     visualize1(learner.network, 1)


plt.show()

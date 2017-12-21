"""
Combine NEF pretraining with FA training on MNIST.

Notes:
- Increasing the learning rate between pretraining and FA causes a huge jump
  in error. This is likely because pretraining has caught on to patterns
  in hidden unit activities that are very sensitive to the particular weights
  (they likely play hidden units off each other). So is FA basically scrapping
  everything that pretraining has done? This doesn't seem useful.
  - Using the same small LR for final layer, larger for earlier FA layers
    seems to help a lot.
  Other ideas for resolution include:
  - more local or regularized weights in initial structure
  - more regularization on pretraining solver
  - larger initial LR on pretraining solver
- Having FA backweights that are aligned with the class selectivities of
  initial weights is necessary.

TODO:
- Need better accuracy from pretraining. Maybe use LR annealing, higher
  momentum, Nesterov momentum (though I had trouble switching from Nesterov
  momentum to normal momentum before).
"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import nengo
import nengo.utils.numpy as npext
from nengo_extras.data import load_mnist, one_hot_from_labels
from nengo_extras.vision import Gabor, Mask, ciw_encoders
# from nengo_extras.vision import Gabor, Mask, ciw_encoders, cd_encoders_biases

import hunse_thesis.solvers
from hunse_thesis.neurons import static_f_df
from hunse_thesis.offline_learning import (
    Network, ShallowLearner, FASkipLearner, make_flat_batch_fn,
    nll_cost_on_inds, class_error_on_inds)
from hunse_thesis.utils import initial_w

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

# --- load data
s_in = (28, 28)
n_in = np.prod(s_in)
n_out = 10

train, test = load_mnist('~/data/mnist.pkl.gz')
# train = (train[0][:1000], train[1][:1000])
# train = (train[0][:10000], train[1][:10000])

trainX, trainY = train
testX, testY = test
for images in [trainX, testX]:
    images[:] = 2 * images - 1  # normalize to -1 to 1

trainT = one_hot_from_labels(trainY, classes=10)
testT = one_hot_from_labels(testY, classes=10)

assert trainX.shape[1] == n_in
assert trainT.shape[1] == n_out


# --- NEF network
# filename = None
# filename = 'combined_offline_mnist_nef.dil'
# filename = 'combined_offline_mnist_test.dil'
# filename = 'combined_offline_mnist_sgd25.dil'
# filename = 'combined_offline_mnist_sgd25vanilla.dil'
# filename = 'combined_offline_mnist_sgd50vanilla.dil'
# filename = 'combined_offline_mnist_sgd50vanillareset.dil'
filename = 'combined_offline_mnist_sgd50nesterov.dil'

if filename is None or not os.path.exists(filename):
    rng = np.random.RandomState(1)

    # neuron_type = nengo.LIF()
    # neuron_type = nengo.LIF(tau_rc=0.02, amplitude=0.025)
    neuron_type = nengo.LIF(tau_rc=0.05, amplitude=0.025)
    n_hids = [1100, 1000]

    print("Encoders")
    # max_rates0 = 100 * np.ones(n_hids[0])
    # intercepts0 = 0.1 * np.ones(n_hids[0])
    # gain0, bias0 = neuron_type.gain_bias(max_rates0, intercepts0)
    gain0 = 1. * np.ones(n_hids[0])
    # gain0 = 0.5 * np.ones(n_hids[0])
    # gain0 = 0.4 * np.ones(n_hids[0])
    bias0 = 1. * np.ones(n_hids[0])
    encoders0 = Mask(s_in).populate(
        Gabor().generate(n_hids[0], (11, 11), rng=rng), rng=rng, flatten=True)
    h0 = neuron_type.rates(np.dot(trainX, encoders0.T), gain0, bias0)

    # max_rates1 = 100 * np.ones(n_hids[1])
    # intercepts1 = 0.1 * np.ones(n_hids[1])
    # gain1, bias1 = neuron_type.gain_bias(max_rates1, intercepts1)
    gain1 = 1. * np.ones(n_hids[1])
    # gain1 = 0.5 * np.ones(n_hids[1])
    # gain1 = 0.4 * np.ones(n_hids[1])
    bias1 = 1. * np.ones(n_hids[1])
    encoders1 = ciw_encoders(n_hids[1], h0, trainY, rng=rng)
    # encoders1 *= Mask(s_in).generate(
    #     n_hid, rf_shape, rng=rng, flatten=True)
    encoders1 /= npext.norm(encoders1, axis=1, keepdims=True)
    h1 = neuron_type.rates(np.dot(h0, encoders1.T), gain1, bias1)

    print("Solving")
    # solver = hunse_thesis.solvers.LstsqClassifier(reg=0.01)
    # solver = hunse_thesis.solvers.Softmax(reg=0.0001)
    # solver = hunse_thesis.solvers.Softmax(reg=0.0001, verbose=1)
    # solver = hunse_thesis.solvers.Softmax(reg=0.001, verbose=1)
    # solver = hunse_thesis.solvers.Softmax(reg=0.001, n_epochs=300, verbose=1)
    # solver = hunse_thesis.solvers.Softmax(reg=0.0001, n_epochs=300, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=1000, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=1000, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=1000, eta=2e-4, momentum=-0.9, verbose=1)

    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=25, eta=1e-3, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=25, eta=2e-4, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=25, eta=1e-4, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=25, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.01, n_epochs=25, momentum=0.5, eta=1e-4, verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=50, momentum=0.5, eta=2e-5, verbose=1)
    solver = hunse_thesis.solvers.SoftmaxSGD(reg=0.001, n_epochs=50, momentum=-0.9, eta=2e-5, verbose=1)
    decoders, solver_info = solver(h1, trainT, rng=rng)

    train_out = np.dot(h1, decoders)
    train_errors = np.argmax(train_out, axis=1) != trainY
    train_error = 100 * train_errors.mean()

    # def get_outs(x):
    #     h0 = neuron_type.rates(np.dot(x, encoders0.T), gain0, bias0)
    #     h1 = neuron_type.rates(np.dot(h0, encoders1.T), gain1, bias1)
    #     y = np.dot(h1, decoders)
    #     return y

    # def get_error(images, labels):
    #     return np.argmax(get_outs(images), axis=1) != labels

    # print("Classifying")
    # train_error = np.mean([
    #     100 * get_error(trainX[10000*i:10000*(i+1)],
    #                     trainY[10000*i:10000*(i+1)]).mean()
    #     for i in range(5)])
    # test_error = 100 * get_error(testX, testY).mean()
    # print("Train/test error: %0.2f%%, %0.2f%%" % (train_error, test_error))

    if filename is not None:
        with open(filename, 'wb') as fh:
            dill.dump(dict(
                neuron_type=neuron_type, n_hids=n_hids,
                gain0=gain0, bias0=bias0, encoders0=encoders0, h0=h0,
                gain1=gain1, bias1=bias1, encoders1=encoders1, h1=h1,
                solver=solver, decoders=decoders,
                train_out=train_out, train_error=train_error,
                ), fh)
        print("Saved %r" % filename)
else:
    with open(filename, 'rb') as fh:
        filedata = dill.load(fh)
        globals().update(filedata)


print("Pretraining solver: %s" % solver)


weights = [encoders0.T * gain0, encoders1.T * gain1, decoders]
biases = [bias0 - 1, bias1 - 1, np.zeros(n_out)]

cost = nll_cost_on_inds
error = class_error_on_inds

f, df = static_f_df('liflinear',
                    tau_rc=neuron_type.tau_rc,
                    tau_ref=neuron_type.tau_ref,
                    amplitude=neuron_type.amplitude)

get_network = lambda **kwargs: Network(
    weights, f=f, df=df, biases=biases, noise=0, **kwargs)

reg = solver.reg
# reg = 0

alpha = (reg * h1.max())**2
# alpha = 1e-5
# alpha = 1e-6
# alpha = 0.00525450331251
# alpha = 0
print("alpha: %s" % alpha)

# network = get_network()
learner = ShallowLearner(
    get_network(), cost, error, eta=0, alpha=alpha)

train_cost = cost(train_out, trainY)[0].mean()
test_error = 100 * learner.test(test).mean()
print("Static train/test error: %0.2e, %0.2f%%, %0.2f%%" % (
    train_cost, train_error, test_error))

# print(", ".join("||W%d|| = %0.3f" % (i, npext.norm(w))
#                 for i, w in enumerate(learner.network.weights)))

# --- try same pretraining solver to make sure cost doesn't jump up
if 0:
    # eta = 1e-4
    # eta = 2e-5
    eta = solver.eta
    momentum = abs(solver.momentum)
    batch_size = solver.batch_size

    solver2 = hunse_thesis.solvers.SoftmaxSGD(
        n_epochs=2, reg=reg, eta=eta, momentum=momentum, batch_size=batch_size,
        verbose=1)
    # solver = hunse_thesis.solvers.SoftmaxSGD(
    #     reg=0.001, n_epochs=1, eta=1e-4, momentum=0.9, batch_size=100, verbose=1)
    # W1 = solver(h1[:100], trainT[:100], X=weights[-1])
    W1, _ = solver2(h1, trainT, X=weights[-1])
    train_out2 = np.dot(h1, W1)
    train_cost2 = cost(train_out2, trainY)[0].mean()
    print(train_cost2)

    batch_fn = make_flat_batch_fn(trainX, trainY, batch_size)

    learner = ShallowLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, momentum=momentum)

    learner.train(2, batch_fn, test_set=test)
    train_out3 = learner.network.predict(trainX)
    train_cost3 = cost(train_out3, trainY)[0].mean()
    train_error3 = error(train_out3, trainY).mean()
    print((train_cost3, train_error3))

    assert 0

# --- try with shallow learner to make sure cost doesn't jump
if 0:
    # eta = 1e-3
    # eta = 1e-4
    eta = 2e-5

    momentum = 0.5

    epochs = 10

    # n_per_batch = 20
    n_per_batch = 100
    batch_fn = make_flat_batch_fn(trainX, trainY, n_per_batch)

    learner = ShallowLearner(
        get_network(), cost, error, eta=eta, alpha=alpha, momentum=momentum)

    learner.train(epochs, batch_fn, test_set=test)
    assert 0


# --- train with FA
rng = np.random.RandomState(2)

# eta = 1e-2
# eta = 4e-3
# eta = 2e-3
# eta = 1e-3
# eta = 5e-4
# eta = 1e-4
# eta = 2e-5
# eta = [2e-4, 2e-4, 2e-5]
eta = [2e-3, 2e-3, 2e-5]

# momentum = 0
momentum = 0.5

epochs = 10

# n_per_batch = 20
n_per_batch = 100
batch_fn = make_flat_batch_fn(trainX, trainY, n_per_batch)

learner = FASkipLearner(
    get_network(), cost, error, eta=eta, alpha=alpha, momentum=momentum)

if 0:
    Bs = [np.zeros((n_out, n_hid)) for n_hid in n_hids]
elif 1:
    Bs = [initial_w((n_out, n_hid), kind='ortho', scale=0.1) for n_hid in n_hids]
else:
    scale = 0.1
    Bs = []
    for h in (h0, h1):
        hmean, hstd = h.mean(axis=0), h.std(axis=0)
        hsilent = hstd <= 1e-16
        h = (h - hmean) / np.maximum(hstd, 1e-16)
        # h = (h - hmean) * (scale / np.maximum(hstd, 1e-16))

        B = np.dot(trainT.T, h) / trainT.sum(0)[:, None]
        if hsilent.sum() > 0:
            Bstd = B[:, ~hsilent].std(axis=0).mean()
            B[:, hsilent] = rng.normal(scale=Bstd, size=(B.shape[0], hsilent.sum()))
            # B[:, hsilent] = initial_w((B.shape[0], hsilent.sum()), kind='gaussian', scale=scale)

        # B *= scale / npext.norm(B, axis=0)
        B *= scale / npext.norm(B, axis=0).mean()

        # print(B[:, :3])
        # print(B.mean(axis=0))
        # print(B.std(axis=0))
        # print(B[:, hsilent].mean(axis=0))
        # print(B[:, hsilent].std(axis=0))
        Bs.append(B)
        # assert 0

learner.Bs = Bs
# learner.Bs = [initial_w((n_out, n_hid), kind='ortho', scale=0.01) for n_hid in n_hids]

learner.train(epochs, batch_fn, test_set=test)

# TODO: look at first layer weights before and after

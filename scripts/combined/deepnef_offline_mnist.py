# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

import nengo
import nengo.utils.numpy as npext
from nengo_extras.data import load_mnist, one_hot_from_labels
from nengo_extras.vision import Gabor, Mask, ciw_encoders
# from nengo_extras.vision import Gabor, Mask, ciw_encoders, cd_encoders_biases

import hunse_thesis.solvers
# from hunse_thesis.vision import (
#     percentile_encoders_intercepts, scale_encoders_intercepts)

rng = np.random.RandomState(1)

# --- load data
s_in = (28, 28)
n_in = np.prod(s_in)
n_out = 10

train, test = load_mnist('~/data/mnist.pkl.gz')
# train = (train[0][:1000], train[1][:1000])
# train = (train[0][:10000], train[1][:10000])

train_images, train_labels = train
test_images, test_labels = test
for images in [train_images, test_images]:
    images[:] = 2 * images - 1  # normalize to -1 to 1

train_targets = one_hot_from_labels(train_labels, classes=10)
test_targets = one_hot_from_labels(test_labels, classes=10)

assert train_images.shape[1] == n_in
assert train_targets.shape[1] == n_out


# --- network
neuron_type = nengo.LIF()
n_hids = [1000, 1000]

print("Encoders")
max_rates0 = 100 * np.ones(n_hids[0])
intercepts0 = 0.1 * np.ones(n_hids[0])
gain0, bias0 = neuron_type.gain_bias(max_rates0, intercepts0)
encoders0 = Mask(s_in).populate(
    Gabor().generate(n_hids[0], (11, 11), rng=rng), rng=rng, flatten=True)
h0 = neuron_type.rates(np.dot(train_images, encoders0.T), gain0, bias0)

max_rates1 = 100 * np.ones(n_hids[1])
intercepts1 = 0.1 * np.ones(n_hids[1])
gain1, bias1 = neuron_type.gain_bias(max_rates1, intercepts1)
encoders1 = ciw_encoders(n_hids[1], h0, train_labels, rng=rng)
# encoders1 *= Mask(s_in).generate(
#     n_hid, rf_shape, rng=rng, flatten=True)
encoders1 /= npext.norm(encoders1, axis=1, keepdims=True)
h1 = neuron_type.rates(np.dot(h0, encoders1.T), gain1, bias1)

print("Solving")
solver = hunse_thesis.solvers.LstsqClassifier(reg=0.01)
decoders, solver_info = solver(h1, train_targets, rng=rng)

train_errors = np.argmax(np.dot(h1, decoders), axis=1) != train_labels
train_error = 100 * train_errors.mean()


def get_outs(x):
    h0 = neuron_type.rates(np.dot(x, encoders0.T), gain0, bias0)
    h1 = neuron_type.rates(np.dot(h0, encoders1.T), gain1, bias1)
    y = np.dot(h1, decoders)
    return y

def get_error(images, labels):
    return np.argmax(get_outs(images), axis=1) != labels


print("Classifying")
# train_error = np.mean([
#     100 * get_error(train_images[10000*i:10000*(i+1)],
#                     train_labels[10000*i:10000*(i+1)]).mean()
#     for i in range(5)])
test_error = 100 * get_error(test_images, test_labels).mean()
print("Train/test error: %0.2f%%, %0.2f%%" % (train_error, test_error))

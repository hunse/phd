"""
Large CIFAR-10 networks that require special code.

NOTES:
Loading data (augment=False)
Encoders (n_hid = 15000, method='gabor')
Train/test error: 7.33%, 42.39%

Loading data (augment=True)
Encoders (n_hid = 15000, method='gabor')
Train/test error: 27.14%, 36.13%
"""
import itertools

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
rng = np.random.RandomState(9)

import nengo
from nengo_extras.data import load_cifar10, one_hot_from_labels
from nengo_extras.vision import Gabor, Mask

# augment = False
augment = True

# n_hid = 5
# n_hid = 100
# n_hid = 500
n_hid = 1000
# n_hid = 2000
# n_hid = 3000
# n_hid = 6000
# n_hid = 12000
# n_hid = 15000
# n_hid = 21000

# --- load the data
print("Loading data (augment=%r)" % (augment,))
train, test = load_cifar10('~/data/cifar-10-python.tar.gz')

train_images, train_labels = train
test_images, test_labels = test
train_images = (1. / 128) * train_images - 1
test_images = (1. / 128) * test_images - 1
del train, test

train_targets = one_hot_from_labels(train_labels, classes=10)
test_targets = one_hot_from_labels(test_labels, classes=10)

assert train_images.ndim == test_images.ndim == 2
n_vis = train_images.shape[1]
n_out = train_targets.shape[1]
shape = (3, 32, 32)
c = shape[0]
assert np.prod(shape) == n_vis

per_batch = 10000
if not augment:
    def batches():
        return itertools.izip(train_images.reshape(-1, per_batch, n_vis),
                              train_targets.reshape(-1, per_batch, n_out))
else:
    batch_images = train_images.reshape((-1, per_batch) + shape)
    batch_targets = train_targets.reshape(-1, per_batch, n_out)
    train_images = train_images.reshape((-1,) + shape)
    test_images = test_images.reshape((-1,) + shape)

    p = 2
    shape = (shape[0], shape[1]-2*p, shape[2]-2*p)
    n_vis = np.prod(shape)
    train_images = train_images[:, :, p:-p, p:-p].reshape(-1, n_vis)
    test_images = test_images[:, :, p:-p, p:-p].reshape(-1, n_vis)

    def batches():
        for _ in range(5):  # number of times through whole dataset
            for images, targets in zip(batch_images, batch_targets):
                i = rng.randint(0, 2*p, size=per_batch)
                j = rng.randint(0, 2*p, size=per_batch)
                cropped = np.array([
                    images[k, :, i[k]:i[k]+shape[1], j[k]:j[k]+shape[2]]
                    for k in range(per_batch)])
                yield cropped.reshape((per_batch, n_vis)), targets

# --- set up network parameters
method = 'gabor'

print("Encoders (n_hid = %d, method=%r)" % (n_hid, method))


if method == 'full':
    encoders = rng.normal(size=(n_hid,) + shape).reshape(n_hid, -1)
elif method == 'mask':
    encoders = Mask(shape).populate(rng.normal(size=(n_hid, c, 9, 9)), rng=rng, flatten=True)
elif method == 'gabor':
    gabors = Gabor().generate(n_hid, (13, 13), rng=rng)
    colors = nengo.dists.UniformHypersphere(surface=True).sample(n_hid, c, rng=rng)
    gabors = gabors[:, None, :, :] * colors[:, :, None, None]
    encoders = Mask(shape).populate(gabors, rng=rng, flatten=True)
else:
    raise ValueError(method)


encoded = np.dot(batches().next()[0], encoders.T)

neuron_type = nengo.LIFRate()
intercepts = np.percentile(encoded, 50, axis=0)
max_rates = 100 * np.ones(n_hid)
gain, bias = neuron_type.gain_bias(max_rates, intercepts)
print("Intercepts: %0.2e (%0.2e)" % (intercepts.mean(), intercepts.std()))

# --- Set up network
def encode(x):
    return neuron_type.rates(np.dot(x, encoders.T), gain, bias)

print("Activities")
AA = np.zeros((n_hid, n_hid))
Ay = np.zeros((n_hid, n_out))
Amax = -np.inf
m = 0

for x, y in batches():
    A = encode(x)
    AA += np.dot(A.T, A)
    Ay += np.dot(A.T, y)
    m += A.shape[0]
    Amax = np.maximum(A.max(), Amax)

print("Solving")
reg = 0.0001

sigma = reg * Amax
np.fill_diagonal(AA, AA.diagonal() + m * sigma**2)
factor = scipy.linalg.cho_factor(AA, overwrite_a=True)
decoders = scipy.linalg.cho_solve(factor, Ay)

def get_outs(images):
    return np.dot(encode(images), decoders)

def get_error(images, labels):
    return np.argmax(get_outs(images), axis=1) != labels

print("Classifying")
train_error = np.mean([
    100 * get_error(train_images[10000*i:10000*(i+1)], train_labels[10000*i:10000*(i+1)]).mean()
    for i in range(5)])
test_error = 100 * get_error(test_images, test_labels).mean()
print("Train/test error: %0.2f%%, %0.2f%%" % (train_error, test_error))

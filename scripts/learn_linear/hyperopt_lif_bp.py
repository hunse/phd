"""

Bests:
- BPLearner:liflinear:40-40
  {amp: 5.4e-03, w_scale: 1.9e-01, w_kind: uniform, eta: 2.5e-01}: 2.257e-01
- BPLearner:liflinear:80-80
  {w_scale: 2.1e-01, neuron_type: {kind: liflinear, amplitude: 4.9e-03}, eta: 1.2e-01, w_kind: uniform}: 1.740e-01
- BPLearner:lifstep:80-80
  {neuron_type: {damplitude: 5.0e+00, amplitude: 4.0e-02, kind: lifstep}, w_kind: gaussian, eta: 5.9e-04, w_scale: 6.6e-02}: 1.758e-01
- BPLearner:liflinear:80-80:lsuv_inputs
  {eta: 1.4e-01, neuron_type: {amplitude: 6.8e-03, kind: liflinear}}: 2.128e-01
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import hyperopt
from hyperopt import hp

from nengo.synapses import Lowpass
from nengo.utils.compat import is_number
from nengo.utils.numpy import norm, rms

from hunse_thesis.neurons import static_f_df, linear, dlinear, relu, drelu
from hunse_thesis.offline_learning import (
    Network, ShallowLearner, BPLearner, FALearner, FASkipLearner, DTPLearner,
    make_flat_batch_fn)
from hunse_thesis.offline_learning import (
    squared_cost, rms_error)

from hunse_thesis.utils import angle, orthogonalize, initial_weights, initial_w, lsuv

def get_vals(trial):
    # based on hyperopt/base.py:Trials:argmin
    return {k: v[0] for k, v in trial['misc']['vals'].items() if v}


# rng = np.random.RandomState(9)
rng = np.random.RandomState(8)
# rng = np.random

din = 30
# dhids = [20]
# dhids = [40]
# dhids = [80]
# dhids = [160]

# dhids = [40, 40]
dhids = [80, 80]

dout = 10
sizes = [din] + dhids + [dout]

n_per_batch = 1
# n_per_batch = 2
# n_per_batch = 5
# n_per_batch = 10
# n_per_batch = 20

# n = 300
# n = 3000
n = 6000
# n = 9000
# n = 12000
# n = 15000
# n = 18000
# n = 25000
# n = 35000
# n = 50000
# n = 80000

n_batches = n // n_per_batch

momentum = 0
# momentum = 0.5

alpha = 0
# alpha = 1e-8
# alpha = 1e-6

# Learner = ShallowLearner
Learner = BPLearner

# --- problem dataset
T = orthogonalize(rng.normal(size=(din, dout)))
genX = lambda n: rng.normal(scale=1., size=(n, din))
genY = lambda X: np.dot(X, T)

X = genX(n)
Y = genY(X)
batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

Xvalid = genX(10000)
Yvalid = genY(Xvalid)

def arg_string(args):
    return "{%s}" % ", ".join(
        "%s: %s" % (k, arg_string(v)) if isinstance(v, dict) else
        "%s: %0.1e" % (k, v) if is_number(v) else
        "%s: %s" % (k, v)
        for k, v in args.items())


def objective(args):
    sargs = arg_string(args)
    w_kind = args['w_kind']
    w_scale = args['w_scale']
    eta = args['eta']
    # eta = [args['eta0'], args['eta1'], args['eta2']]

    max_cost = -np.inf
    for _ in range(5):
        f, df = static_f_df(tau_rc=0.05, **args['neuron_type'])

        weights = initial_weights(sizes, kind=w_kind, scale=w_scale, rng=rng)
        # weights = initial_weights(sizes, kind='ortho', rng=rng)
        # lsuv(X, weights, f, target_input=True, target_std=1, verbose=1)
        # lsuv(X[:100], weights, f, target_input=True, target_std=1, verbose=1)

        # --- learners
        network = Network(weights, f=f, df=df, biases=None)
        # network = Network(weights, f=f, df=df, biases=None, noise=1.)
        learner = Learner(network, squared_cost, rms_error,
                          eta=eta, alpha=alpha, momentum=momentum)
        learner.train(1, batch_fn, verbose=0)

        y = learner.network.predict(Xvalid)
        mean_cost = rms(y - Yvalid, axis=1).mean()
        max_cost = max(max_cost, mean_cost)

    print("%s: %0.3e" % (sargs, max_cost))

    return max_cost


r = np.log(10)
space = {
    'neuron_type': hp.choice('neuron_type', [
        {'kind': 'liflinear',
         'amplitude': hp.lognormal('amp0', -2*r, 1*r)},
        # {'kind': 'lifstep',
        #  'amplitude': hp.lognormal('amp1', -2*r, 1*r),
        #  'damplitude': hp.lognormal('damp1', -1*r, 1*r)}
        ]),
    'w_kind': hp.choice('w_kind', ['ortho', 'gaussian', 'uniform']),
    'w_scale': hp.lognormal('w_scale', -2*r, 1*r),
    'eta': hp.lognormal('eta', -2*r, 1*r),
    # 'eta0': hp.lognormal('eta0', -3*r, 1*r),
    # 'eta1': hp.lognormal('eta1', -3*r, 1*r),
    # 'eta2': hp.lognormal('eta2', -3*r, 1*r),
}

trials = hyperopt.Trials()
fmin_args = dict(algo=hyperopt.tpe.suggest, trials=trials)
# hyperopt.fmin(objective, space, max_evals=1, **fmin_args)
# hyperopt.fmin(objective, space, max_evals=2, **fmin_args)
# hyperopt.fmin(objective, space, max_evals=5, **fmin_args)
hyperopt.fmin(objective, space, max_evals=100, **fmin_args)
# hyperopt.fmin(objective, space, max_evals=200, **fmin_args)
# hyperopt.fmin(objective, space, max_evals=500, **fmin_args)

best_trial = trials.best_trial
best_args = hyperopt.space_eval(space, get_vals(best_trial))
best_loss = best_trial['result']['loss']

print("Best: %s: %0.3e" % (arg_string(best_args), best_loss))

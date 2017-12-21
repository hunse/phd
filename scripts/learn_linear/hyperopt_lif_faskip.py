"""

Bests:
- FASkipLearner:30-80-80-10
  {neuron_type: {amplitude: 2.4e-02, kind: liflinear},
  b_kind: gaussian, b_scale: 1.7e+00, eta: 4.4e-03,
  w_kind: gaussian, w_scale: 4.4e-04, alpha: 3.9e-08}: 1.773e-01

- FASkipLearner:30-80-80-10, b_normkind=rightmean
  {neuron_type: {amplitude: 3.8e-03, kind: liflinear}, b_kind: uniform,
  w_kind: uniform, eta: 5.7e-01, w_scale: 1.7e-02}: 1.671e-01

"""
import warnings
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

warnings.filterwarnings('ignore', message="invalid value encountered")
warnings.filterwarnings('ignore', message="overflow encountered")


def get_vals(trial):
    # based on hyperopt/base.py:Trials:argmin
    return {k: v[0] for k, v in trial['misc']['vals'].items() if v}

def arg_string(args):
    return "{%s}" % ", ".join(
        "%s: %s" % (k, arg_string(v)) if isinstance(v, dict) else
        "%s: %0.1e" % (k, v) if is_number(v) else
        "%s: %s" % (k, v)
        for k, v in args.items())


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

Learner = FASkipLearner

# --- problem dataset
T = orthogonalize(rng.normal(size=(din, dout)))
# genX = lambda n: rng.normal(scale=1., size=(n, din))
genX = lambda n: rng.normal(scale=0.5, size=(n, din))
genY = lambda X: np.dot(X, T)

X = genX(n)
Y = genY(X)
batch_fn = make_flat_batch_fn(X, Y, n_per_batch)

Xvalid = genX(10000)
Yvalid = genY(Xvalid)
Yvalidrms = rms(Yvalid, axis=1).mean()

r = np.log(10)
space = {
    'neuron_type': hp.choice('neuron_type', [
        {'kind': 'liflinear',
         'amplitude': hp.lognormal('amp0', -2*r, 1*r)},
        ]),
    'w_kind': hp.choice('w_kind', ['ortho', 'gaussian', 'uniform']),
    'w_scale': hp.lognormal('w_scale', -2*r, 1*r),
    'b_kind': hp.choice('b_kind', ['ortho', 'gaussian', 'uniform']),
    # 'b_scale': hp.lognormal('b_scale', 0*r, 1*r),
    'eta': hp.lognormal('eta', -2*r, 1*r),
    # 'alpha': hp.lognormal('alpha', -6*r, 2*r),
}


def objective(args):
    sargs = arg_string(args)
    w_kind = args['w_kind']
    w_scale = args['w_scale']
    b_kind = args['b_kind']
    # b_scale = args['b_scale']
    eta = args['eta']
    # alpha = args['alpha']
    alpha = 0
    # eta = [args['eta0'], args['eta1'], args['eta2']]

    # max_cost = -np.inf
    costs = []
    for _ in range(5):
        f, df = static_f_df(tau_rc=0.05, **args['neuron_type'])

        weights = initial_weights(sizes, kind=w_kind, scale=w_scale, rng=rng)

        # --- learners
        network = Network(weights, f=f, df=df, biases=None)
        # network = Network(weights, f=f, df=df, biases=None, noise=1.)
        learner = Learner(network, squared_cost, rms_error, eta=eta, alpha=alpha)
        # learner.Bs = [initial_w((dout, dhid), kind=b_kind, scale=b_scale) for dhid in dhids]
        learner.Bs = [initial_w((dout, dhid), kind=b_kind, normkind='rightmean') for dhid in dhids]

        learner.train(1, batch_fn, verbose=0)

        y = learner.network.predict(Xvalid)
        cost = rms(y - Yvalid, axis=1).mean() / Yvalidrms
        costs.append(cost)

    costs = sorted(costs)[1:-1]  # drop largest and smallest
    cost = np.mean(costs)
    status = hyperopt.STATUS_OK if np.isfinite(cost) else hyperopt.STATUS_FAIL
    print("%s: %0.3e" % (sargs, cost))

    return dict(loss=cost, status=status)


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

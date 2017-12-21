"""

Bests:
"""
import datetime
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import hyperopt
from hyperopt import hp

import nengo
from nengo.dists import Uniform
from nengo_extras.dists import Tile
from nengo.utils.compat import is_number
from nengo.utils.numpy import norm, rms

from hunse_thesis.online_learning import (
    Encoder, ShallowNetwork, FASkipNetwork, FATwoStepNetwork)

from hunse_thesis.hyperopt import get_vals, wrap_cost
from hunse_thesis.utils import initial_weights, orthogonalize

def arg_string(args):
    return "{%s}" % ", ".join(
        "%s: %s" % (k, arg_string(v)) if isinstance(v, dict) else
        "%s: %0.1e" % (k, v) if is_number(v) else
        "%s: %s" % (k, v)
        for k, v in args.items())

def eye_encoders(d):
    return Tile(np.vstack((np.eye(d), -np.eye(d))))


warnings.filterwarnings('ignore', message='This learning rate is very high')


rng = np.random.RandomState(8)

# --- parameters
din = 30

# dhids = [40, 40]
dhids = [80, 80]

dout = 10
sizes = [din] + dhids + [dout]

# t_train = 2
# t_train = 10
t_train = 500

synapse = nengo.synapses.Alpha(0.003)
neuron_type = nengo.LIF(tau_rc=0.05, amplitude=0.024)

# --- saving
s_sizes = '-'.join('%d' % s for s in sizes)
s_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filedir = 'results/hyperopt_online_%s_%s' % (s_sizes, s_now)
os.mkdir(filedir)

# --- space
r = np.log(10)
space = {
    'eta': hp.lognormal('eta', np.log10(5e-2), 1*r),
    'prestime': hp.lognormal('prestime', 0*r, 0.2*r),
}


def objective(args):
    eta = args['eta']
    prestime = args['prestime']

    n_examples = int(np.ceil(t_train / prestime))

    # dataset
    T = orthogonalize(rng.normal(size=(din, dout)))
    genX = lambda n: rng.normal(scale=0.5, size=(n, din))
    genY = lambda X: np.dot(X, T)
    X = genX(n_examples)
    Y = genY(X)
    x_process = nengo.processes.PresentInput(X, prestime)
    y_process = nengo.processes.PresentInput(Y, prestime)

    Xtest = genX(1000)
    Ytest = genY(Xtest)
    Ytestrms = rms(Ytest, axis=1).mean()

    weights = initial_weights([2*din] + dhids + [dout],
                              kind='gaussian', scale=5e-4, rng=rng)

    fa_args = dict(eta=eta, seed=2, n_output=20, n_error=20,
                   o_encoders=eye_encoders(dout), e_encoders=eye_encoders(dout),
                   e_intercepts=Uniform(0, 0.8),
                   b_kind='gaussian', b_scale=1.7)

    model = nengo.Network()
    model.config[nengo.Ensemble].neuron_type = neuron_type
    model.config[nengo.Connection].synapse = synapse
    with model:
        x = nengo.Node(x_process)
        y = nengo.Node(y_process)
        xe = Encoder(x, seed=1)
        learner = FATwoStepNetwork(xe.output, y, weights, **fa_args)
        xp = nengo.Probe(x)
        yp = nengo.Probe(y)
        xep = nengo.Probe(xe.output)

    with nengo.Simulator(model) as sim:
        sim.run(t_train)

        XEtest = xe.encode(Xtest, sim=sim)
        Ztest = learner.forward(sim, XEtest)
        cost = rms(Ztest - Ytest, axis=1).mean() / Ytestrms

    # save
    dt = sim.dt
    t = sim.trange()
    x = sim.data[xp]
    xe = sim.data[xep]
    y = sim.data[yp]
    z = sim.data[learner.yp]
    data = dict(eta=eta, prestime=prestime, dt=dt, t_train=t_train,
                din=din, dhids=dhids, dout=dout,
                T=T, X=X, Y=Y, Xtest=Xtest, Ytest=Ytest, XEtest=XEtest, Ztest=Ztest,
                # t=t, x=x, xe=xe, y=y, z=z,
                cost=cost)

    rargs = '_'.join('%s=%r' % (k, v) for k, v in args.items())
    filename = os.path.join(filedir, 'trial_%s.npz' % (rargs))
    np.savez(filename, **data)
    print("Saved %r" % filename)

    # result
    status = hyperopt.STATUS_OK if np.isfinite(cost) else hyperopt.STATUS_FAIL
    sargs = arg_string(args)
    print("%s: %0.3e" % (sargs, cost))

    return dict(loss=cost, status=status)


trials = hyperopt.Trials()
fmin_args = dict(algo=hyperopt.tpe.suggest, trials=trials)
safe_objective = wrap_cost(objective, timeout=20*60, iters=2, verbose=1)
# hyperopt.fmin(safe_objective, space, max_evals=1, **fmin_args)
# hyperopt.fmin(safe_objective, space, max_evals=2, **fmin_args)
# hyperopt.fmin(safe_objective, space, max_evals=5, **fmin_args)
# hyperopt.fmin(safe_objective, space, max_evals=50, **fmin_args)
hyperopt.fmin(safe_objective, space, max_evals=80, **fmin_args)

best_trial = trials.best_trial
best_args = hyperopt.space_eval(space, get_vals(best_trial))
best_loss = best_trial['result']['loss']
print("Best: %s: %0.3e" % (arg_string(best_args), best_loss))

# --- save trials file
trialsfile = os.path.join(filedir, 'trials.npz')
dtrials = []
for trial in trials.trials:
    vals = get_vals(trial)
    args = hyperopt.space_eval(space, vals)
    dtrial = dict(vals=vals, args=args, result=trial['result'])
np.savez(trialsfile, trials=dtrials)
print("Saved %r" % trialsfile)

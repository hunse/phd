"""
Get the Lillicrap et al random backprop method working on their simple example.

NOTES:
- Output derivative might not be working as well because it takes some time
  to shut off neuron learning. So as we switch to the next stimulus, if the
  neuron becomes inactive, it still learns for a while afterwards. Also,
  if the threshold is too high, that results in only intermittent learning
  when spike rates are low.
  What about learning that's triggered whenever there's a spike?
- To re-create refractory IF derivative based on output current,
  what if there was a finite resource that increased exponentially,
  and was used as part of learning? Or it could be if neuron is very active
  there is a lot of forward current, this depleats resources in dendrites
  and inhibits backpropagating APs.
"""
import datetime
import sys

import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.numpy import norm, rms
from nengo_extras.dists import Tile

from hunse_thesis.online_learning import (
    Encoder, ShallowNetwork, FASkipNetwork, FATwoStepNetwork)
from hunse_thesis.utils import initial_weights, orthogonalize

def eye_encoders(d):
    return Tile(np.vstack((np.eye(d), -np.eye(d))))

np.random.seed(1)
# rng = np.random.RandomState(9)
rng = np.random.RandomState(8)
# rng = np.random

# din = 6
din = 30

# dhid = 20
# dhid = 40
# dhid = 80
# dhid = 160

# dhids = [20, 20]
# dhids = [40, 40]
dhids = [80, 80]

# dout = 2
dout = 10

if 1:
    # t_train = 10
    t_train = 500
    # t_train = 1000

    # prestime = 0.1
    # prestime = 0.2
    # prestime = 0.22
    prestime = 0.25
    # prestime = 0.333
    # prestime = 0.5
    # prestime = 1.0
    # prestime = 5.0
    # prestime = 20.0
    # prestime = 100.0

    # n_test_pre = 0
    # n_test_pre = 1
    # n_test_pre = 5
    # n_test_pre = 10
    n_test_pre = 20

    # eta = 0.9
    # eta = 0.5
    # eta = 0.25
    # eta = 0.1
    eta = 5e-2
    # eta = 2.3e-2
    # eta = 2e-2
    # eta = 1.6e-2
    # eta = 1e-2
    # eta = 2e-3
    # eta = 5e-3
else:
    # t_train = 5
    # prestime = 2.5
    # n_test_pre = 0

    t_train = 20
    prestime = 0.25
    # prestime = 2.5
    # prestime = 5
    n_test_pre = 0

    eta = 0.1
    # eta = 1.6e-2

n_train = int(np.ceil(t_train / prestime))
n_test_post = max(int(0.1 * n_train), 1)
n_examples = n_test_pre + n_train + n_test_post

t0 = n_test_pre*prestime
t1 = t0 + n_train*prestime
t2 = t1 + n_test_post*prestime


# --- normalized transform
T = orthogonalize(rng.normal(size=(din, dout)))

if 1:
    # genX = lambda n: rng.normal(scale=1., size=(n, din))
    genX = lambda n: rng.normal(scale=0.5, size=(n, din))
    genY = lambda X: np.dot(X, T)
    X = genX(n_examples)
    Y = genY(X)
    x_process = nengo.processes.PresentInput(X, prestime)
    y_process = nengo.processes.PresentInput(Y, prestime)
else:
    genX = lambda n: rng.normal(scale=0.5, size=(n, din))
    genY = lambda X: np.dot(X, T)
    freq = (0.5/np.pi) / prestime  # angular frequency = 1. / prestime
    p = nengo.processes.WhiteSignal(n_examples * prestime, freq, rms=0.5)
    X = p.run(n_examples * prestime, d=din, dt=0.001, rng=rng)
    Y = np.dot(X, T)
    x_process = lambda t: X[int(t / 0.001) % len(X)]
    y_process = lambda t: Y[int(t / 0.001) % len(Y)]

Xtest = genX(1000)
Ytest = genY(Xtest)


# weights = initial_weights([2*din] + dhids + [dout], kind='ortho', rng=rng)
# weights = initial_weights([2*din] + dhids + [dout], kind='uniform', scale=0.2, rng=rng)
# weights = initial_weights([2*din] + dhids + [dout], kind='uniform', scale=0.4, rng=rng)
weights = initial_weights([2*din] + dhids + [dout], kind='gaussian', scale=4.4e-4, rng=rng)
# weights = initial_weights([2*din] + dhids + [dout], kind='zeros')

# neuron_type = nengo.LIF(tau_rc=0.05, amplitude=0.005)
neuron_type = nengo.LIF(tau_rc=0.05, amplitude=0.024)
# synapse = nengo.synapses.Alpha(0.005)
synapse = nengo.synapses.Alpha(0.003)
# synapse_n = lambda n: reduce(lambda a, b: a.combine(b), [synapse] * n)

model = nengo.Network()
model.config[nengo.Ensemble].neuron_type = neuron_type
model.config[nengo.Connection].synapse = synapse
# network_args = dict(t0=t0, t1=t1, eta=eta, seed=2)
# network_args = dict(n_output=20, n_error=20, t0=t0, t1=t1, eta=eta, seed=2)
# network_args = dict(n_output=40, n_error=40, t0=t0, t1=t1, eta=eta, seed=2)
# network_args = dict(t0=t0, t1=t1, eta=eta, seed=2, n_output=40, n_error=40)
# network_args = dict(t0=t0, t1=t1, eta=eta, seed=2, n_output=40, n_error=40,
#                     o_kind='array', e_kind='array')

# network_args = dict(t0=t0, t1=t1, eta=eta, seed=2, n_output=20, n_error=20,
#                     o_encoders=eye_encoders(dout), e_encoders=eye_encoders(dout),
#                     e_intercepts=Uniform(0, 0.8))

n = 2
network_args = dict(t0=t0, t1=t1, eta=eta, seed=2,
                    n_output=2*n*dout, n_error=2*n*dout,
                    o_encoders=eye_encoders(dout), e_encoders=eye_encoders(dout),
                    e_intercepts=Uniform(0, 0.8))


fa_args = dict(network_args)
fa_args.update(dict(b_kind='gaussian', b_scale=1.7))

with model:
    x = nengo.Node(x_process)
    xp = nengo.Probe(x)

    ystar = nengo.Node(y_process)
    ystarp = nengo.Probe(ystar)

    # --- encode x in spiking neurons
    xenc = Encoder(x, seed=1)
    xencp = nengo.Probe(xenc.output)

    # --- learners
    learners = []

    # learners.append(ShallowNetwork(xenc.output, ystar, weights, **network_args))

    # learners.append(FASkipNetwork(xenc.output, ystar, weights, **fa_args))

    fa2 = FATwoStepNetwork(xenc.output, ystar, weights, **fa_args)
    fa2.hps = [nengo.Probe(h.neurons) for h in fa2.layers]
    learners.append(fa2)


# XE = xenc.encode(X)
# shallow.scale_weights(XE)
# faskip.scale_weights(XE)
# print(XE.min(), XE.max(), XE.std())
# assert 0


# with nengo.Simulator(model, optimize=False) as sim:

import nengo_ocl
with nengo_ocl.Simulator(model) as sim:

    sim.run(t1)
    # sim.run(t2)

    Ytestrms = rms(Ytest, axis=1).mean()
    XEtest = xenc.encode(Xtest, sim=sim)
    for learner in learners:
        y = learner.forward(sim, XEtest)
        learner.Ytest = y
        e = rms(y - Ytest, axis=1).mean() / Ytestrms
        print("%s: %0.3e" % (learner, e))

    sim.run(t2 - t1)

# --- save results
s_sizes = '-'.join('%d' % s for s in [din] + dhids + [dout])
s_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = 'results/online_linear_%s_t=%s_eta=%0.1e_%s.npz' % (
    s_sizes, t2, eta, s_now)

dt = sim.dt
t = sim.trange()
x = sim.data[xp]
xe = sim.data[xencp]
ystar = sim.data[ystarp]
keys = ['n_test_pre', 'n_train', 'n_test_post', 'prestime', 'dt', 'eta',
        'din', 'dhids', 'dout', 'T', 'X', 'Y', 'Xtest', 'Ytest', 'XEtest',
        't', 'x', 'xe', 'ystar']
data = dict((k, globals()[k]) for k in keys)

d_learners = []
for learner in learners:
    d_learner = dict(
        name=str(learner), y=sim.data[learner.yp], e=sim.data[learner.ep],
        Ytest=learner.Ytest)
    if isinstance(learner, FATwoStepNetwork):
        d_learner['els'] = [sim.data[elp] for elp in learner.elps]
    if hasattr(learner, 'hps'):
        d_learner['hs'] = [sim.data[hp].astype('float32') for hp in learner.hps]
    d_learners.append(d_learner)
data['learners'] = d_learners

if t_train >= 50:
    np.savez(filename, **data)
    print("Saved %r" % filename)

from plot_online_linear import show_all_plots
show_all_plots(data)

"""
"""
import datetime
import warnings

import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.numpy import rms
from nengo_extras.data import load_mnist, one_hot_from_labels
from nengo_extras.dists import Tile

from hunse_thesis.online_learning import (
    Encoder, ShallowNetwork, FASkipNetwork, FATwoStepNetwork)
from hunse_thesis.utils import initial_weights

from plot_online_mnist import show_all_plots, print_test_errors

def eye_encoders(d):
    return Tile(np.vstack((np.eye(d), -np.eye(d))))

warnings.filterwarnings('ignore', message='This learning rate is very high')

# rng = np.random.RandomState(9)
rng = np.random.RandomState(8)
# rng = np.random

# --- data
mnist = load_mnist('~/data/mnist.pkl.gz')
(Xtrain, Ytrain), (Xtest, Ytest) = mnist

labels = np.unique(Ytrain)
n_labels = len(labels)

def preprocess(images):
    images[:] *= 2
    images[:] -= 1

preprocess(Xtrain), preprocess(Xtest)
Ttrain = one_hot_from_labels(Ytrain, classes=n_labels)
Ttest = one_hot_from_labels(Ytest, classes=n_labels)

# --- params
# dhids = [600, 300]
dhids = [500, 500]

if 1:
    # epochs = 0.1
    # epochs = 0.5
    # epochs = 1.5
    # epochs = 3
    epochs = 6
    # epochs = 10
    # epochs = 15

    # prestime = 0.2
    prestime = 0.33

    # eta = 1.
    # eta = 0.5
    # eta = 0.25
    # eta = 0.1
    # eta = 5e-2
    # eta = 2e-2
    eta = 1e-2
    # eta = 5e-3
    # eta = 2e-3

    t_train = prestime * Xtrain.shape[0] * epochs
    t_test = prestime * 100
else:
    epochs = 0
    # prestime = 1.0
    prestime = 0.33

    # eta = 1.
    # eta = 0.5
    # eta = 0.25
    eta = 0.1
    # eta = 5e-2
    # eta = 2e-2
    # eta = 1e-2
    # eta = 2e-3
    # eta = 5e-3

    t_train = 5.
    t_test = 0.001

t0 = 0
t1 = t0 + t_train
t2 = t1 + t_test

din = Xtrain.shape[1]
dout = n_labels
sizes = [din] + dhids + [dout]

# --- model
weights = initial_weights([2*din] + dhids + [dout], kind='uniform', scale=4e-3, rng=rng)
# weights = initial_weights([2*din] + dhids + [dout], kind='uniform', scale=4e-8, rng=rng)

# neuron_type = nengo.LIF(tau_rc=0.05, amplitude=0.0014)
neuron_type = nengo.LIF(tau_rc=0.05, amplitude=0.024)
synapse = nengo.synapses.Alpha(0.003)

model = nengo.Network()
model.config[nengo.Ensemble].neuron_type = neuron_type
model.config[nengo.Connection].synapse = synapse

pdt = 0.01

network_args = dict(t0=t0, t1=t1, eta=eta, seed=2, n_output=20, n_error=20,
                    o_encoders=eye_encoders(dout), e_encoders=eye_encoders(dout),
                    e_intercepts=Uniform(0, 0.8), pdt=pdt)

fa_args = dict(network_args)
fa_args.update(dict(b_kind='gaussian', b_normkind='rightmean'))
fa_args.update(dict(b_scale=1.7))

with model:
    x = nengo.Node(nengo.processes.PresentInput(Xtrain, prestime))
    xp = nengo.Probe(x, sample_every=pdt)

    ystar = nengo.Node(nengo.processes.PresentInput(Ttrain, prestime))
    ystarp = nengo.Probe(ystar, sample_every=pdt)

    # --- encode x in spiking neurons
    xenc = Encoder(x, seed=1)
    xencp = nengo.Probe(xenc.output, sample_every=pdt)

    # --- learners
    learners = []
    # learners.append(ShallowNetwork(xenc.output, ystar, weights, **network_args))
    # learners.append(FASkipNetwork(xenc.output, ystar, weights, **fa_args))
    learners.append(FATwoStepNetwork(xenc.output, ystar, weights, **fa_args))

    # fa2 = FATwoStepNetwork(xenc.output, ystar, weights, **fa_args)
    # fa2.hps = [nengo.Probe(h.neurons) for h in fa2.layers]
    # learners.append(fa2)

# with nengo.Simulator(model, optimize=False) as sim:
import nengo_ocl
# with nengo_ocl.Simulator(model) as sim:
with nengo_ocl.Simulator(model, progress_bar=False) as sim:
    sim.run(t1)

    Ttestrms = rms(Ttest, axis=1).mean()
    XEtest = xenc.encode(Xtest, sim=sim)
    for learner in learners:
        learner.Ttest = learner.forward(sim, XEtest)
        e = (np.argmax(learner.Ttest, axis=1) != Ytest).mean()
        print("%s: %0.2e" % (learner, 100 * e))

    sim.run(t2 - t1)

# --- save results
s_sizes = '-'.join('%d' % s for s in [din] + dhids + [dout])
s_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = 'results/online_mnist_%s_t=%s_eta=%0.1e_%s.npz' % (
    s_sizes, t2, eta, s_now)

dt = sim.dt
t = sim.trange(pdt)
# x = sim.data[xp]
# xe = sim.data[xencp]
ystar = sim.data[ystarp]
# keys = ['din', 'dhids', 'dout', 'epochs', 'prestime', 'dt', 'pdt', 'eta',
#         't_train', 't_test', 'XEtest',
#         't', 'x', 'xe', 'ystar']
keys = ['din', 'dhids', 'dout', 'epochs', 'prestime', 'dt', 'pdt', 'eta',
        't_train', 't_test', 'XEtest', 't', 'ystar']
data = dict((k, globals()[k]) for k in keys)

d_learners = []
for learner in learners:
    d_learner = dict(
        name=str(learner), y=sim.data[learner.yp], e=sim.data[learner.ep],
        Ttest=learner.Ttest)
    if isinstance(learner, FATwoStepNetwork):
        d_learner['els'] = [sim.data[elp] for elp in learner.elps]
    d_learners.append(d_learner)
data['learners'] = d_learners

np.savez(filename, **data)
print("Saved %r" % filename)

show_all_plots(data, mnist)

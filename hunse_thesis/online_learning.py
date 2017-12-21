"""
Learning in spiking neurons with Nengo
"""
import numpy as np

import nengo
from nengo.dists import Choice, Gaussian, Uniform, UniformHypersphere
from nengo.learning_rules import PES
from nengo.neurons import Linear
from nengo.utils.compat import is_iterable

from nengo_extras.dists import Tile
from nengo_extras.learning_rules import DeltaRule

from .neurons import deltarule_df, dliflinear
from .utils import EAIO, initial_w, neural_activities


class Encoder(nengo.Network):
    """Encode x in on-off spiking neurons
    """
    def __init__(self, x, intercepts=Choice([0]), max_rates=Uniform(100, 120),
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # encode x in spiking neurons
        d = x.size_out

        with self:
            self.output = nengo.Ensemble(
                2*d, d, intercepts=intercepts, max_rates=max_rates,
                label='encoder')
        nengo.Connection(x, self.output.neurons[:d], transform=1, synapse=None)
        nengo.Connection(x, self.output.neurons[d:], transform=-1, synapse=None)

    def encode(self, x, sim=None):
        if x.ndim == 1:
            x = x.reshape((1, -1))
        y = np.column_stack((x, -x))
        if sim is None:
            with nengo.Simulator(self) as sim:
                z = neural_activities(sim, self.output, y)
        else:
            z = neural_activities(sim, self.output, y)
        return z


class FeedforwardNetwork(nengo.Network):

    def __init__(self, x, ystar, initial_weights, t0=None, t1=None,
                 n_output=None, o_kind='ensemble',
                 o_encoders=UniformHypersphere(surface=True),
                 o_intercepts=Uniform(-1, 0.9), o_rates=Uniform(100, 120),
                 n_error=None, e_kind='ensemble',
                 e_encoders=UniformHypersphere(surface=True),
                 e_intercepts=Uniform(-1, 0.9), e_rates=Uniform(100, 120),
                 psynapse=None, pdt=None, **kwargs):
        super(FeedforwardNetwork, self).__init__(**kwargs)

        din = initial_weights[0].shape[0]
        dout = initial_weights[-1].shape[1]
        dhids = [w.shape[1] for w in initial_weights[:-1]]

        self.n_output = n_output
        self.n_error = n_error
        self.e_kind = e_kind
        self.e_encoders = e_encoders
        self.e_intercepts = e_intercepts
        self.e_rates = e_rates

        self.pargs = dict(synapse=psynapse, sample_every=pdt)

        with self:
            # hidden layers
            self.layers = [
                nengo.Ensemble(dhid, dhid,
                               gain=Choice([1.]), bias=Choice([1.]),
                               label='layer%d' % i)
                for i, dhid in enumerate(dhids)]

            # output layer
            if n_output is None:
                self.output = EAIO(
                    nengo.Node, size_in=dout, label='output')
            elif o_kind == 'ensemble':
                self.output = EAIO(
                    nengo.Ensemble, n_output*dout, dout, label='output',
                    encoders=o_encoders, intercepts=o_intercepts, max_rates=o_rates)
            elif o_kind == 'array':
                self.output = nengo.networks.EnsembleArray(
                    n_output, dout, label='output',
                    encoders=o_encoders, intercepts=o_intercepts, max_rates=o_rates)

            self.yp = nengo.Probe(self.output.output, **self.pargs)

            # error layer
            tmask = ((lambda t: (t > t0) & (t < t1)) if t0 is not None and t1 is not None else \
                     (lambda t: (t > t0)) if t0 is not None else \
                     (lambda t: (t < t1)) if t1 is not None else \
                     (lambda t: (t >= 0)))
            nmask = lambda t: 20 * (tmask(t) - 1)
            if n_error is None:
                ferror = lambda t, x: tmask(t) * x
                self.error = EAIO(
                    nengo.Node, ferror, size_in=dout, label='error')
            elif e_kind == 'ensemble':
                self.error = EAIO(
                    nengo.Ensemble, n_error*dout, dout, label='error',
                    encoders=e_encoders, intercepts=e_intercepts, max_rates=e_rates)
                error_switch = nengo.Node(nmask)
                nengo.Connection(error_switch, self.error.neuron_input,
                                 transform=np.ones((n_error*dout, 1)), synapse=None)
            elif e_kind == 'array':
                self.error = nengo.networks.EnsembleArray(
                    n_error, dout, label='error',
                    encoders=e_encoders, intercepts=e_intercepts, max_rates=e_rates)
                self.error.add_neuron_input()
                error_switch = nengo.Node(nmask)
                nengo.Connection(error_switch, self.error.neuron_input,
                                  transform=np.ones((n_error*dout, 1)), synapse=None)
            else:
                raise ValueError(e_kind)
            self.ep = nengo.Probe(self.error.output, **self.pargs)

        # connections
        nengo.Connection(ystar, self.error.input, transform=-1)
        with self:
            nengo.Connection(self.output.output, self.error.input)

        self.conns = []
        self.conns.append(nengo.Connection(
            x.neurons, self.layers[0].neurons, transform=initial_weights[0].T))

        with self:
            layers = self.layers
            # layers = self.layers + [self.output]
            for layer0, layer1, w in zip(layers, layers[1:], initial_weights[1:]):
                self.conns.append(nengo.Connection(
                    layer0.neurons, layer1.neurons, transform=w.T))
            self.conns.append(nengo.Connection(
                self.layers[-1].neurons, self.output.input, transform=initial_weights[-1].T))

    def __str__(self):
        return type(self).__name__

    def forward(self, sim, x):
        for c, e in zip(self.conns, self.layers + [None]):
            w = sim.signals[sim.model.sig[c]['weights']]
            x = np.dot(x, w.T)
            if isinstance(e, nengo.Ensemble):
                x = neural_activities(sim, e, x)
            elif isinstance(e, nengo.Node):
                assert e.output is None
            else:
                assert e is None

        return x

    def scale_weights(self, x, std=0.5):
        n = 100
        din = self.conns[0].size_in
        # x = rng.uniform(-1, 1, size=(n, din))
        with nengo.Simulator(self) as sim:
            for c, e in zip(self.conns, self.layers + [None]):
                y = np.dot(x, c.transform.T)
                ystd = y.std()
                print(ystd)
                c.transform = c.transform * (std / ystd)

                if e is not None:
                    y = np.dot(x, c.transform.T)
                    x = neural_activities(sim, e, y)


class ShallowNetwork(FeedforwardNetwork):
    def __init__(self, *args, **kwargs):
        eta = kwargs.pop('eta', 1e-2)
        super(ShallowNetwork, self).__init__(*args, **kwargs)

        c = self.conns[-1]
        # c.learning_rule_type = DeltaRule(learning_rate=eta, post_fn=step)
        c.learning_rule_type = PES(learning_rate=eta)
        nengo.Connection(self.error.output, c.learning_rule)


class FASkipNetwork(FeedforwardNetwork):
    def __init__(self, *args, **kwargs):
        eta = kwargs.pop('eta', 1e-2)
        b_kind = kwargs.pop('b_kind', 'ortho')
        b_normkind = kwargs.pop('b_normkind', None)
        b_scale = kwargs.pop('b_scale', 1.)
        super(FASkipNetwork, self).__init__(*args, **kwargs)

        dout = self.output.output.size_out

        # step = lambda x: (x > 1).astype(x.dtype)
        # ^ TODO: need x to be seriously positive, since filtered value will never be 0.
        #     BUT it's on the input, not the output!

        # --- backwards connections
        for c in self.conns[:-1]:
            neuron_type = c.post_obj.ensemble.neuron_type
            assert isinstance(neuron_type, nengo.LIF)
            def df(j, a=neuron_type.amplitude, tau_rc=neuron_type.tau_rc,
                   tau_ref=neuron_type.tau_ref):
                return a * dliflinear(j, tau_rc, tau_ref)

            c.learning_rule_type = DeltaRule(learning_rate=eta, post_fn=df)

            B = initial_w((dout, c.post.size_in),
                          kind=b_kind, normkind=b_normkind, scale=b_scale)
            nengo.Connection(self.error.output, c.learning_rule, transform=B.T)

        c = self.conns[-1]
        # c.learning_rule_type = DeltaRule(learning_rate=lr)
        c.learning_rule_type = PES(learning_rate=eta)
        nengo.Connection(self.error.output, c.learning_rule)

        # self.conns[0].learning_rule_type.pre_tau = 0.0  # filtered before input


class FATwoStepNetwork(FeedforwardNetwork):
    """
    Need layer errors to go to zero when master error is zero. This should
    happen by default with NEF ensembles, but it means we always need to use
    encoders and decoders with these ensembles.
    """
    def __init__(self, *args, **kwargs):
        eta = kwargs.pop('eta', 1e-2)
        b_kind = kwargs.pop('b_kind', 'ortho')
        b_normkind = kwargs.pop('b_normkind', None)
        b_scale = kwargs.pop('b_scale', 1.)
        e_kind = kwargs.pop('e_kind', 'ensemble')
        super(FATwoStepNetwork, self).__init__(*args, **kwargs)

        dout = self.output.output.size_out
        dhids = [x.n_neurons for x in self.layers]
        n_error = self.n_error

        with self:
            # --- backwards error layers
            labels = ['elayer%d' % i for i in range(len(self.layers))]
            if n_error is None:
                self.elayers = [EAIO(nengo.Node, size_in=dout, label=label)
                                for label in labels]
            elif self.e_kind == 'ensemble':
                self.elayers = [EAIO(
                    nengo.Ensemble, n_error*dout, dout, label=label,
                    encoders=self.e_encoders, intercepts=self.e_intercepts,
                    max_rates=self.e_rates)
                                for label in labels]
            elif e_kind == 'array':
                self.elayers = [nengo.networks.EnsembleArray(
                    n_error, dout, label=label, encoders=self.e_encoders,
                    intercepts=self.e_intercepts, max_rates=self.e_rates)
                                for label in labels]

            self.elps = [nengo.Probe(elayer.output, **self.pargs)
                         for elayer in self.elayers]

        # --- backwards (deep) connections
        error_layers = [self.error] + self.elayers[::-1]
        for e0, e1 in zip(error_layers, error_layers[1:]):
            nengo.Connection(e0.output, e1.input)
            # nengo.Connection(e0.output, e1.input, transform=initial_w((dout, dout), kind='ortho'))

        for c, e in zip(self.conns, self.elayers):
            neuron_type = c.post_obj.ensemble.neuron_type

            # print("Derivative on neuron input")
            # assert isinstance(neuron_type, nengo.LIF)
            # def df(j, a=neuron_type.amplitude, tau_rc=neuron_type.tau_rc,
            #        tau_ref=neuron_type.tau_ref):
            #     return a * dliflinear(j, tau_rc, tau_ref)

            # c.learning_rule_type = DeltaRule(learning_rate=eta, post_fn=df)

            print("Derivative on neuron input")
            post_target = 'in'
            post_tau = 0.005
            # post_fn = deltarule_df('liflinear', neuron_type, post_target=post_target)
            # post_fn = deltarule_df('step', neuron_type, post_target=post_target, damplitude=0.5)
            # post_fn = deltarule_df('step', neuron_type, post_target=post_target, damplitude=0.25)
            post_fn = deltarule_df('step', neuron_type, post_target=post_target, damplitude=0.33)

            # print("Derivative on neuron output")
            # post_target = 'out'
            # post_tau = 0.005
            # # post_fn = deltarule_df('step', neuron_type, post_target=post_target,
            # #                        threshold=np.exp(-6)/post_tau)
            # #                        # threshold=np.exp(-4)/post_tau)
            # post_fn = deltarule_df('step', neuron_type, post_target=post_target,
            #                        threshold=np.exp(-6)/post_tau, damplitude=0.33)

            c.learning_rule_type = DeltaRule(
                learning_rate=eta, post_target=post_target, post_fn=post_fn,
                post_tau=post_tau)

            B = initial_w((dout, c.post.size_in),
                          kind=b_kind, normkind=b_normkind, scale=b_scale)
            nengo.Connection(e.output, c.learning_rule, transform=B.T)

        # --- output (shallow) connection
        c = self.conns[-1]
        # c.learning_rule_type = DeltaRule(learning_rate=lr)
        c.learning_rule_type = PES(learning_rate=eta)
        nengo.Connection(self.error.output, c.learning_rule)

        # self.conns[0].learning_rule_type.pre_tau = 0.0  # filtered before input


# class DTPNetwork(FeedforwardNetwork):
#     def __init__(self, *args, **kwargs):
#         super(DTPNetwork, self).__init__(*args, **kwargs)

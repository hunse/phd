import numpy as np

import nengo
from nengo.builder import Builder, Signal
from nengo.builder.neurons import SimNeurons
from nengo.params import NumberParam

from nengo_extras.neurons import softplus


def epsilon(dtype):
    return np.nextafter(np.array(0, dtype=dtype), 1)


def linear(x):
    return x


def dlinear(x):
    return np.ones_like(x)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def dsigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)


def sigmoid1(x):
    return 1. / (1 + np.exp(-4*x))


def dsigmoid1(x):
    y = sigmoid1(x)
    return 4 * y * (1 - y)


def relu(x):
    return np.maximum(x, 0)


def drelu(x):
    return (x > 0).astype(x.dtype)


def lif1p(j, tau_rc, tau_ref):
    return 1. / (tau_ref + tau_rc*np.log1p(1./np.maximum(j, 0)))


def lif(j, tau_rc, tau_ref):
    return 1. / (tau_ref + tau_rc*np.log1p(1./np.maximum(j - 1, 0)))


def dlif(j, tau_rc, tau_ref):
    y = lif(j, tau_rc, tau_ref)
    return (tau_rc * y * y) / np.maximum(j * (j - 1), epsilon(j.dtype))


def lifinv(r, tau_rc, tau_ref):
    er = np.expm1((1./np.maximum(r, 0) - tau_ref) / tau_rc)
    return 1 + 1./np.maximum(er, 0)


def liflinear(j, tau_rc, tau_ref):
    return 1. / (tau_ref + tau_rc/np.maximum(j - 1, 0))


def dliflinear(j, tau_rc, tau_ref):
    return (j > 1) * tau_rc / (tau_ref*np.maximum(j - 1, 0) + tau_rc)**2


def softlif(j, tau_rc, tau_ref, sigma=1.):
    j = softplus(j - 1, sigma=sigma)
    return 1. / (tau_ref + tau_rc*np.log1p(1./j))


def dsoftlif(j, tau_rc, tau_ref, sigma=1.):
    y = j - 1
    j = softplus(y, sigma=sigma)
    v = lif1p(j, tau_rc, tau_ref)
    d = np.zeros_like(j)
    yy, vv, jj = y[j > 0], v[j > 0], j[j > 0]
    # computing numerator and denominator separately is more numerically stable
    d[j > 0] = (tau_rc*vv*vv) / (jj*(jj + 1)*(1 + np.exp(-yy/sigma)))
    return d


def softliflinear(j, tau_rc, tau_ref, sigma=1.):
    return 1. / (tau_ref + tau_rc/softplus(j - 1, sigma=sigma))


def lifleak(j, tau_rc, tau_ref, leak=1.):
    if leak < 1e-8:
        return 1. / (tau_ref + tau_rc/np.maximum(j, 0))
    else:
        return 1. / (tau_ref + (tau_rc/leak)*np.log1p(leak/np.maximum(j - leak, 0)))


def static_f_df(kind, **kwargs):
    tau_rc = kwargs.get('tau_rc', 0.05)
    tau_ref = kwargs.get('tau_ref', 0.002)
    a = kwargs.get('amplitude', 1./63)

    kind = kind.lower()
    if kind == 'linear':
        return linear, dlinear
    elif kind == 'sigmoid':
        return sigmoid, dsigmoid
    elif kind == 'sigmoid1':
        return sigmoid1, dsigmoid1
    elif kind == 'relu':
        return relu, drelu
    elif kind == 'softlif':
        from nengo_extras.neurons import SoftLIFRate
        sigma = kwargs.get('sigma', 0.02)
        softlif = SoftLIFRate(tau_rc=tau_rc, tau_ref=tau_ref,
                              amplitude=a, sigma=sigma)
        f = lambda x: softlif.rates(x, 1., 1.)
        df = lambda x: softlif.derivative(x, 1., 1.)
        return f, df
    elif kind == 'lifsoftlif':
        from nengo_extras.neurons import SoftLIFRate
        sigma = kwargs.get('sigma', 0.02)
        softlif = SoftLIFRate(tau_rc=tau_rc, tau_ref=tau_ref, sigma=sigma)
        f = lambda x: a * lif(x + 1, tau_rc, tau_ref)
        df = lambda x: a * softlif.derivative(x, 1., 1.)
        return f, df
    elif kind == 'lifnone':
        d = kwargs.get('damplitude', 1.)
        f = lambda x: a * lif(x + 1, tau_rc, tau_ref)
        df = lambda x: (d * a / tau_rc) * np.ones_like(x)
        return f, df
    elif kind == 'lifact':  # "derivative" equal to forward activity
        d = kwargs.get('damplitude', 1.)
        clip = kwargs.get('clip', None)
        f = lambda x: a * lif(x + 1, tau_rc, tau_ref)
        if clip is None:
            df = lambda x: (d * a) * lif(x + 1, tau_rc, tau_ref)
        else:
            df = lambda x: (d * a) * lif(x + 1, tau_rc, tau_ref).clip(None, clip / tau_rc)
        return f, df
    elif kind == 'liflinear':
        d = kwargs.get('damplitude', 1.)
        f = lambda x: a * lif(x + 1, tau_rc, tau_ref)
        df = lambda x: (d * a) * dliflinear(x + 1, tau_rc, tau_ref)
        return f, df
    elif kind == 'lifstep':
        d = kwargs.get('damplitude', 1.)
        f = lambda x: a * lif(x + 1, tau_rc, tau_ref)
        df = lambda x: (d * a / tau_rc) * (x > 0).astype(x.dtype)
        return f, df
    elif kind == 'lifclip':
        clip = kwargs.get('clip', 1.)
        f = lambda x: a * lif(x + 1, tau_rc, tau_ref)
        df = lambda x: a * dlif(x + 1, tau_rc, tau_ref).clip(None, clip / tau_rc)
        return f, df
    else:
        raise ValueError("Unrecognized type %r" % kind)


def deltarule_df(kind, neuron_type, damplitude=1., threshold=1., post_target='in'):
    assert isinstance(neuron_type, (nengo.LIF, nengo.LIFRate))
    a = neuron_type.amplitude
    d = damplitude
    tau_rc = neuron_type.tau_rc
    tau_ref = neuron_type.tau_ref

    if kind == 'step':
        if post_target == 'in':
            df = lambda j: (d * a / tau_rc) * (j > 1).astype(j.dtype)
        elif post_target == 'out':
            df = lambda y: (d * a / tau_rc) * (y > a*threshold).astype(y.dtype)
    elif kind == 'liflinear':
        if post_target == 'in':
            df = lambda j: (d * a) * dliflinear(j, tau_rc, tau_ref)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Unrecognized post function kind %r" % kind)

    return df


class IFRate(nengo.neurons.NeuronType):
    """Non-spiking version of the integrate-and-fire (IF) neuron model.

    Parameters
    ----------
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    """

    probeable = ('rates',)

    tau_ref = NumberParam('tau_ref', low=0)
    amplitude = NumberParam('amplitude', low=0)

    def __init__(self, tau_ref=0., amplitude=1.):
        super(IFRate, self).__init__()
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    @property
    def _argreprs(self):
        args = []
        if self.tau_ref != 0.:
            args.append("tau_ref=%s" % self.tau_ref)
        if self.amplitude != 1.:
            args.append("amplitude=%s" % self.amplitude)
        return args

    def gain_bias(self, max_rates, intercepts):
        """Determine gain and bias by shifting and scaling the lines."""
        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def rates(self, x, gain, bias):
        """Always use IFRate to determine rates."""
        J = gain * x + bias
        out = np.zeros_like(J)
        # Use IFRate's step_math explicitly to ensure rate approximation
        IFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Implement the IFRate nonlinearity."""
        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = self.amplitude / (self.tau_ref + 1. / j[j > 0])
        # the above line is designed to throw an error if any j is nan
        # (nan > 0 -> error), and not pass x < -1 to log1p


class IF(IFRate):
    """Spiking version of the integrate-and-fire (IF) neuron model.

    Parameters
    ----------
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    """

    probeable = ('spikes', 'voltage', 'refractory_time')

    min_voltage = NumberParam('min_voltage', high=0)

    def __init__(self, tau_ref=0., amplitude=1., min_voltage=0):
        super(IF, self).__init__(tau_ref=tau_ref, amplitude=amplitude)
        self.min_voltage = min_voltage

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = (dt - refractory_time).clip(0, dt)

        # update voltage by integrating
        voltage += delta_t * J

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt - (voltage[spiked_mask] - 1) / J[spiked_mask]

        # set spiked neuron refractory times to tau_ref
        refractory_time[spiked_mask] = self.tau_ref + t_spike

        # set spiked neuron voltages to zero, unless the ref period is small
        delta_t2 = (dt - refractory_time[spiked_mask]).clip(0, dt)
        voltage[spiked_mask] = delta_t2 * J[spiked_mask]

        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage

@Builder.register(IF)
def build_if(model, neuron_type, neurons):
    """Builds a `.IF` object into a model. """
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.add_op(SimNeurons(
        neurons=neuron_type,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time']]))

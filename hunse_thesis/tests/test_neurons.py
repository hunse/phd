import pytest

import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.utils.numpy import rms

from hunse_thesis.neurons import *


def test_sigmoid1():
    x, dx = np.linspace(-6, 6, 101, retstep=True)
    y = sigmoid1(x)
    plt.subplot(211)
    plt.plot(x, y)
    plt.subplot(212)
    dy = np.diff(y) / dx
    plt.plot(0.5*(x[:-1] + x[1:]), dy, 'k')
    plt.plot(x, dsigmoid1(x))
    plt.show()


def test_liflinear():
    # kwargs = dict(tau_rc=0.05, tau_ref=0.002)
    kwargs = dict(tau_rc=0.02, tau_ref=0.002, amplitude=1.)

    x, dx = np.linspace(-5, 30, 1001, retstep=True)

    f, df = static_f_df('softlif', sigma=0.2, **kwargs)
    y0 = f(x)
    d0 = df(x)

    f, df = static_f_df('liflinear', **kwargs)
    y1 = f(x)
    d1 = df(x)

    a = kwargs.pop('amplitude')
    y2 = a * lif(x + 1, **kwargs)
    d2 = np.diff(y2) / dx
    x2 = 0.5 * (x[:-1] + x[1:])

    plt.subplot(211)
    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.plot(x, y2, '--')

    plt.subplot(212)
    plt.plot(x, d0)
    plt.plot(x, d1)
    # plt.plot(x2, d2.clip(0, 100))

    plt.show()


def test_lifinv():
    kwargs = dict(tau_rc=0.02, tau_ref=0.002)

    x, dx = np.linspace(-5, 30, 1001, retstep=True)
    y = lif(x, **kwargs)
    z = lifinv(y, **kwargs)

    plt.plot(x, x)
    plt.plot(x, z)

    # # y = np.linspace(-1, 510)
    # y = np.linspace(-1, 5)
    # z = lifinv(y, **kwargs)
    # plt.plot(y, z)
    # # plt.loglog(y, z)

    plt.show()


def test_lifstep():
    # kwargs = dict(tau_rc=0.05, tau_ref=0.002)
    kwargs = dict(tau_rc=0.02, tau_ref=0.002)

    x = np.linspace(-5, 30, 101)

    f, df = static_f_df('softlif', **kwargs)
    y0 = f(x)
    d0 = df(x)

    f, df = static_f_df('lifstep', damplitude=0.5, **kwargs)
    y1 = f(x)
    d1 = df(x)

    plt.subplot(211)
    plt.plot(x, y0)
    plt.plot(x, y1)

    plt.subplot(212)
    plt.plot(x, d0)
    plt.plot(x, d1)

    plt.show()


@pytest.mark.parametrize('dt', (0.001, 0.002))
def test_if(rng, dt):
    """Test that the dynamic model approximately matches the rates."""
    n = 5000
    x = 0.5
    encoders = np.ones((n, 1))
    max_rates = rng.uniform(low=10, high=200, size=n)
    intercepts = rng.uniform(low=-1, high=1, size=n)

    neuron_type = IF()
    # neuron_type = IF(tau_ref=0.002)

    m = nengo.Network()
    with m:
        ins = nengo.Node(x)
        ens = nengo.Ensemble(
            n, dimensions=1, neuron_type=neuron_type,
            encoders=encoders, max_rates=max_rates, intercepts=intercepts)
        nengo.Connection(
            ins, ens.neurons, transform=np.ones((n, 1)), synapse=None)
        spike_probe = nengo.Probe(ens.neurons)
        voltage_probe = nengo.Probe(ens.neurons, 'voltage')
        ref_probe = nengo.Probe(ens.neurons, 'refractory_time')

    t_final = 1.0
    with nengo.Simulator(m, dt=dt) as sim:
        sim.run(t_final)

    i = 3
    plt.subplot(311)
    plt.plot(sim.trange(), sim.data[spike_probe][:, :i])
    plt.subplot(312)
    plt.plot(sim.trange(), sim.data[voltage_probe][:, :i])
    plt.subplot(313)
    plt.plot(sim.trange(), sim.data[ref_probe][:, :i])
    plt.ylim([-dt, ens.neuron_type.tau_ref + dt])
    # plt.show()

    # check rates against analytic rates
    math_rates = ens.neuron_type.rates(
        x, *ens.neuron_type.gain_bias(max_rates, intercepts))
    spikes = sim.data[spike_probe]
    sim_rates = (spikes > 0).sum(0) / t_final
    me = (sim_rates - math_rates).mean()
    rrmse = rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20)

    print("ME = %f" % me)
    print("RRMSE = %f" % rrmse)

    print(sim_rates[:10])
    print(math_rates[:10])

    assert np.sum(math_rates > 0) > 0.5 * n, (
        "At least 50% of neurons must fire")
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)

    # if voltage and ref time are non-constant, the probe is doing something
    assert np.abs(np.diff(sim.data[voltage_probe])).sum() > 1
    assert np.abs(np.diff(sim.data[ref_probe])).sum() > 1

    # compute spike counts after each timestep
    actual_counts = (spikes > 0).cumsum(axis=0)
    expected_counts = np.outer(sim.trange(), math_rates)
    assert (abs(actual_counts - expected_counts) < 1).all()

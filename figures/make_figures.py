import matplotlib.pyplot as plt
import numpy as np


# sigma = 0.2
# sigma = 1.0
sigma = 0.01
tau_ref = 0.002
tau_rc = 0.03
# amp = 20. * tau_ref
amp = 1.
# alpha = 2.
alpha = 1.


def logistic(x):
    y = np.zeros_like(x)
    m = abs(x) < 30.
    y[m] = 1. / (1. + np.exp(-x[m]))
    y[x >= 30.] = 1.
    return y


def softrelu(x, sigma=1.):
    y = x / sigma
    z = np.array(x)
    z[y < 34.0] = sigma * np.log1p(np.exp(y[y < 34.0]))
    return z
    # ^ 34.0 gives exact answer in 32 or 64 bit but doesn't overflow in 32 bit


def d_softrelu(x, sigma=1.):
    return logistic(x / sigma)


def lif_j(j):
    v = np.zeros_like(j)
    ji = 1. / j[j > 0]
    v[j > 0] = amp / (tau_ref + tau_rc * np.log1p(ji))
    return v


def d_lif_j(j):
    v = lif_j(j)
    d = np.zeros_like(j)
    vv, jj = v[j > 0], j[j > 0]
    d[j > 0] = tau_rc * vv * vv / (amp * jj * (jj + 1))
    return d


def lif(x):
    return lif_j(alpha * (x - 1))


def d_lif(x):
    return alpha * d_lif_j(alpha * (x - 1))


def softlif(x, sigma=1.):
    return lif_j(softrelu(alpha * (x - 1), sigma=sigma))


def d_softlif(x, sigma=1.):
    y = alpha * (x - 1)
    j = softrelu(y, sigma=sigma)
    # return d_lif_j(j) * d_softrelu(y, sigma=sigma) * alpha

    # computing numerator and denominator separately is more numerically stable
    v = lif_j(j)
    d = np.zeros_like(j)
    yy, vv, jj = y[j > 0], v[j > 0], j[j > 0]
    d[j > 0] = (alpha * tau_rc * vv * vv) / (amp * jj * (jj + 1) * (1 + np.exp(-yy / sigma)))
    return d


def make_softlif():

    # x = np.linspace(-0.5, 0.5, 10001).astype('float64')
    x = np.linspace(0.5, 1.5, 10001).astype('float64')

    # y = softrelu(x, sigma=sigma)
    # dy = d_softrelu(x, sigma=sigma)
    y = lif(x)
    dy = d_lif(x)
    z = softlif(x, sigma=sigma)
    dz = d_softlif(x, sigma=sigma)

    diffx = 0.5 * (x[:-1] + x[1:])
    diffy = np.diff(y) / (x[1] - x[0])
    diffz = np.diff(z) / (x[1] - x[0])

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.plot(x, y, 'k')
    plt.plot(x, z, 'k--')
    # plt.xlim([-0.5, 0.5])
    plt.xlim([0.5, 1.5])
    plt.ylim([-1.5, 30])
    plt.xlabel('input current ($j$)')
    plt.ylabel('firing rate ($r$) [Hz]')
    plt.legend(['LIF', 'soft LIF'], loc=2)

    plt.subplot(122)
    plt.plot(x, dy, 'k')
    plt.plot(x, dz, 'k--')
    # plt.xlim([-0.5, 0.5])
    plt.xlim([0.5, 1.5])
    plt.ylim([-10, 300])
    plt.xlabel('input current ($j$)')
    plt.ylabel('firing rate derivative ($dr / dj$)')

    plt.tight_layout()
    plt.savefig('softlif.pdf')

    # plt.plot(diffx, diffy)
    # plt.plot(diffx, diffz)

    # plt.show()

def make_spikenoise():
    import nengo

    xmin = 0
    xmax = 10
    tend = 10.0

    tau_rc = 0.02
    tau_ref = 0.004
    neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)

    # synapse = nengo.synapses.Lowpass(0.005)
    # synapse = nengo.synapses.Alpha(0.005)
    synapse = nengo.synapses.Alpha(0.003)
    # synapse = nengo.synapses.Alpha(0.01)

    with nengo.Network() as model:
        n = 50
        a = nengo.Ensemble(n, 1)
        a.gain = np.zeros(n)
        a.bias = np.linspace(xmin, xmax, n)
        jp = nengo.Probe(a.neurons, 'input')
        rp = nengo.Probe(a.neurons, synapse=synapse)

    # sim = nengo.Simulator(model, dt=1e-3)
    sim = nengo.Simulator(model, dt=1e-4)
    sim.run(tend)

    t = sim.trange()
    tmin = 1.0
    x = sim.data[jp][t > tmin]
    y = sim.data[rp][t > tmin]

    assert all(x.std(0) < 1e-8)

    x = x.mean(0)
    y_mean = y.mean(0)
    y_50 = np.percentile(y, 50, axis=0)
    y_25_75 = np.array([y_50 - np.percentile(y, 25, axis=0),
                        np.percentile(y, 75, axis=0) - y_50])
    y_min_max = np.array([y_mean - y.min(0), y.max(0) - y_mean])

    print y[:, x > 1].std(0).mean()
    print y[:, x > 5].std(0).mean()

    # analytical curve
    # y_ref = neuron_type.rates(x, 1., 0.)

    # plt.figure(figsize=(7.0, 5.5))
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(x, y_mean, 'k-')
    plt.plot(x, y_50, 'kx')
    eb = plt.errorbar(x, y_mean, y_min_max, fmt=None, ecolor='k')
    eb[-1][0].set_linestyle(':')
    plt.errorbar(x, y_50, y_25_75, fmt=None, ecolor='k')

    plt.xlabel('input current ($j$)')
    plt.ylabel('filtered response')
    plt.tight_layout()

    plt.savefig('noise.pdf')
    # plt.show()


def make_gabors():
    from nengo.dists import Uniform
    from nengo_extras.matplotlib import tile
    from nengo_extras.vision import Gabor
    from hunse_thesis.dists import LogUniform

    rng = np.random.RandomState(3)

    # r, c = 10, 20
    r, c = 9, 12

    # gabor = Gabor()
    # gabor = Gabor(freq=Uniform(0.5, 1.5))
    gabor = Gabor(freq=LogUniform(np.log(0.5), np.log(1.5)))

    gabors = gabor.generate(r*c, (32, 32), rng=rng)

    tile(gabors, rows=r, cols=c)

    plt.savefig('gabors.pdf')
    # plt.show()


if __name__ == '__main__':
    make_softlif()
    # make_spikenoise()
    # make_gabors()

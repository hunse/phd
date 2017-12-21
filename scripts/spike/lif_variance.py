import matplotlib.pyplot as plt
import nengo
import numpy as np

import seaborn as sns
sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

rng = np.random

# neuron = nengo.LIF(tau_rc=0.02)
neuron = nengo.LIF(tau_rc=0.05, tau_ref=0.005)
neuron_rates = lambda j: neuron.rates(j, gain=1, bias=0)


class Lowpass(object):
    def __init__(self, tau):
        self.tau = tau
        self.synapse = nengo.synapses.Lowpass(tau)
        self.impulse = lambda t: (1/tau)*np.exp(-t/tau)

    def latex(self):
        return r"exponential ($\tau=%s$)" % (self.tau)

    def filtered_segment(self, p, t01):
        tau = self.tau
        p = p[:, None]
        t = p * t01

        # geometric series
        a = (1./tau) * np.exp(-t/tau)
        r = np.exp(-p/tau)
        s = a / (1 - r)

        s[p.ravel() > 1e5] = 0
        return s


class Alpha(object):
    def __init__(self, tau):
        self.tau = tau
        self.synapse = nengo.synapses.Alpha(tau)
        self.impulse = lambda t: (t/tau**2)*np.exp(-t/tau)

    def latex(self):
        return r"alpha ($\tau=%s$)" % (self.tau)

    def filtered_segment(self, p, t01):
        tau = self.tau
        p = p[:, None]
        t = p * t01

        # arithmetico-geometric series
        a = t
        b = (1./tau**2) * np.exp(-t/tau)
        d = p
        r = np.exp(-p/tau)
        s = b * (a / (1 - r) + d*r / (1 - r)**2)

        s[p.ravel() > 1e5] = 0
        return s


def plot_variance(synapse, ylim=None, ax=None):
    if ax is None:
        ax = plt.gca()

    # j = np.linspace(0, 5, 51)
    # j = np.linspace(0, 10, 51)
    j = np.linspace(0, 20, 51)

    r = neuron_rates(j)
    p = 1./r
    t01 = np.linspace(0, 1, 101)

    s = synapse.filtered_segment(p, t01)
    x = j
    y = s

    y_mean = y.mean(axis=1)
    y_50 = np.percentile(y, 50, axis=1)
    # y_25_75 = np.array([y_50 - np.percentile(y, 25, axis=1),
    #                     np.percentile(y, 75, axis=1) - y_50])
    # y_min_max = np.array([y_mean - y.min(axis=1), y.max(axis=1) - y_mean])
    y_25 = np.percentile(y, 25, axis=1)
    y_75 = np.percentile(y, 75, axis=1)
    y_min = y.min(axis=1)
    y_max = y.max(axis=1)

    palette = sns.color_palette()
    c = palette.pop(0)

    ax.plot(x, y_mean, 'k')
    ax.plot(x, y_50, '--', color=c)
    ax.fill_between(x, y_min, y_max, color=c, alpha=0.3)
    ax.fill_between(x, y_25, y_75, color=c, alpha=0.4)
    if ylim:
        plt.ylim(ylim)

    plt.xlabel('input current ($j$)')
    plt.ylabel('filtered response')
    plt.title(synapse.latex())


def hide_axes(ax, x=False, y=False):
    if x:
        ax.set_xlabel('')
        ax.set_xticklabels([])

    if y:
        ax.set_ylabel('')
        ax.set_yticklabels([])


def plot_variances():
    plt.figure(figsize=(6.35, 5.0))
    # plt.figure(figsize=(7.0, 5.5))

    # ylim = (0, 350)
    ylim0 = (0, 350)
    ylim1 = (0, 170)

    rows, cols = 2, 2
    ax = plt.subplot(rows, cols, 1)
    plot_variance(Lowpass(0.003), ylim=ylim0, ax=ax)
    hide_axes(ax, x=True)

    ax = plt.subplot(rows, cols, 2)
    plot_variance(Lowpass(0.005), ylim=ylim0, ax=ax)
    hide_axes(ax, x=True, y=True)

    ax = plt.subplot(rows, cols, 3)
    plot_variance(Alpha(0.003), ylim=ylim1, ax=ax)

    ax = plt.subplot(rows, cols, 4)
    plot_variance(Alpha(0.005), ylim=ylim1, ax=ax)
    hide_axes(ax, y=True)

    plt.tight_layout()

    plt.savefig('lif_variance.pdf')
    plt.show()


if __name__ == '__main__':
    plot_variances()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import seaborn as sns

from hunse_thesis.neurons import lif, liflinear, relu, softlif

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

# matplotlib.rc('text', usetex=True)


def derivative_plot():
    tau_rc = 0.05
    # tau_rc = 0.02

    tau_ref = 0.002
    # tau_ref = 0.005
    # tau_ref = 0.02

    amp = 1.
    stepamp = amp / tau_rc

    n = 1000

    # x, dx = np.linspace(0, 1, n+1, retstep=True)
    # x, dx = np.linspace(0, 5, n+1, retstep=True)
    # x, dx = np.linspace(0, 10, n+1, retstep=True)
    x, dx = np.linspace(0, 20, n+1, retstep=True)

    y0 = lif(x, tau_rc, tau_ref)
    y1 = stepamp * relu(x - 1)
    y2 = liflinear(x, tau_rc, tau_ref)
    y3 = softlif(x, tau_rc, tau_ref, sigma=0.146)
    # y3 = lif(x, tau_rc, 0)

    ys = [y0, y1, y2, y3]
    # labels = ['refractory LIF', 'IF', 'refractory IF']
    labels = ['LIF ($t_\mathrm{ref} = %0.3f$)' % tau_ref,
              'IF ($t_\mathrm{ref} = 0$)',
              'IF ($t_\mathrm{ref} = %0.3f$)' % tau_ref,
              'soft-LIF ($t_\mathrm{ref} = %0.3f$)' % tau_ref,
    ]

    x2 = 0.5*(x[:-1] + x[1:])
    diff = lambda y: (np.diff(y) / dx).clip(None, 50)
    dys = [diff(y) for y in ys]


    plt.figure(figsize=(6.4, 5))
    # rows, cols = 1, 2
    rows, cols = 2, 2

    plt.subplot(rows, cols, 1)
    [plt.plot(x, y, label=label) for y, label in zip(ys, labels)]
    plt.xlabel('input current $j$')
    plt.ylabel('firing rate $r$ [Hz]')
    # plt.legend(loc='best')
    # plt.legend(loc=2)

    plt.subplot(rows, cols, 2)
    [plt.plot(x2, dy, label=label) for dy, label in zip(dys, labels)]
    plt.ylabel('firing rate derivative $dr/dj$')
    plt.xlabel('input current $j$')
    plt.legend(loc=1)

    if rows >= 2:  # show zoom in on initial area
        m = x < 2
        m2 = x2 < 2

        plt.subplot(rows, cols, 3)
        [plt.plot(x[m], y[m], label=label) for y, label in zip(ys, labels)]
        plt.xlabel('input current $j$')
        plt.ylabel('firing rate $r$ [Hz]')

        plt.subplot(rows, cols, 4)
        [plt.plot(x2[m2], dy[m2], label=label) for dy, label in zip(dys, labels)]
        plt.ylabel('firing rate derivative $dr/dj$')
        plt.xlabel('input current $j$')

    sns.despine()
    plt.tight_layout()

    plt.savefig('derivative.pdf')


derivative_plot()
plt.show()

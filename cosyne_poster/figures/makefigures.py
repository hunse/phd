import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import seaborn as sns
sns.set_style('white')

from hunse_thesis.neurons import lif, liflinear, dliflinear

# matplotlib.rc('text', usetex=True)


def lifderivative_plot():
    tau_rc = 0.05
    # tau_rc = 0.02

    tau_ref = 0.005
    # tau_ref = 0.02

    n = 1000

    # x, dx = np.linspace(0, 1, n+1, retstep=True)
    # x, dx = np.linspace(0, 5, n+1, retstep=True)
    x, dx = np.linspace(0, 10, n+1, retstep=True)
    # x, dx = np.linspace(0, 20, n+1, retstep=True)

    y = lif(x, tau_rc, tau_ref)
    y1 = liflinear(x, tau_rc, tau_ref)

    # y1 = 1. / (tau_ref + tau_rc/np.maximum(x, 0))
    # y1 = 1. / (tau_ref + tau_rc/np.maximum(x - 0.5, 0))

    # dy1 = dliflinear(x, tau_rc, tau_ref)



    # plt.figure(figsize=(4, 6))
    # rows, cols = 2, 1
    plt.figure(figsize=(5.5, 3))
    rows, cols = 1, 2

    plt.subplot(rows, cols, 1)
    plt.plot(x, y, label='refractory LIF**')
    plt.plot(x, y1, label='refractory IF')

    plt.xlabel('input current $j$')
    plt.ylabel('firing rate $r$ [Hz]')
    # plt.legend(loc='best')
    plt.legend(loc=2)

    plt.subplot(rows, cols, 2)
    x2 = 0.5*(x[:-1] + x[1:])
    plt.plot(x2, (np.diff(y) / dx).clip(None, 50), label='refractory LIF')
    plt.plot(x2, np.diff(y1) / dx, label='refractory IF**')

    plt.ylabel('firing rate derivative $dr/dj$')
    plt.xlabel('input current $j$')
    plt.legend(loc=1)

    plt.tight_layout()

    # plt.plot(x, dy1)

    # plt.plot(x, x)
    # plt.plot(x, np.log(1 + x))

    plt.savefig('lifderivative.pdf')


lifderivative_plot()
plt.show()

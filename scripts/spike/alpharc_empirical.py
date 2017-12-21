"""See how well max and min of alpha-RC series output fits with empirical

If these fit well with the empirically-RC-filtered alpha-train output,
then this should work quite well for training, since the ultimate (t -> inf)
value represents well what happens as the membrane voltage increases.

NOTES:
- They fit well for high firing rates, but not for low rates. This means that
for low rates, the output is still a large spike, and can thus have a strong
occasional influence on downstream neurons (rather than a weak constant one
as the rate-neuron equation describes).

"""
import matplotlib.pyplot as plt
import numpy as np
import nengo
import seaborn as sns

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')

# tau_rc = 0.02
tau_rc = 0.05
tau_s = 0.003

# tmax = 0.1
tmax = 0.15

dt = 0.0001


def alpha_series(p, t):
    t = t % p

    q_s = np.exp(-t/tau_s)
    r_s1 = -np.expm1(-p/tau_s)  # 1 - exp(-p/tau_s)
    tau_r_s1 = tau_s * r_s1
    return q_s*(t + (p - t)*(1 - r_s1)) / (tau_r_s1*tau_r_s1)


def alpharc_series(p, t):
    d = tau_rc - tau_s
    t = t % p

    q_rc = np.exp(-t/tau_rc)
    q_s = np.exp(-t/tau_s)
    r_rc1 = -np.expm1(-p/tau_rc)  # 1 - exp(-p/tau_rc)
    r_s1 = -np.expm1(-p/tau_s)  # 1 - exp(-p/tau_s)
    return (tau_rc/d**2)*q_rc/r_rc1 - (tau_rc/d**2)*q_s/r_s1 - (
        q_s/(tau_s*d)*(t/r_s1 + p*(1-r_s1)/r_s1**2))


def alpha_alpharc_plot():
    # p = 0.01
    p = 0.05

    t = dt * np.arange(int(tmax / dt))
    y0 = alpha_series(p, t)[0]
    y1 = alpharc_series(p, t)[0]

    membrane = nengo.synapses.Lowpass(tau_rc)

    y0f = membrane.filt(y0, y0=0, dt=dt)
    # y1f = membrane.filt(y1, y0=0)

    expt = -np.expm1(-t/tau_rc)
    y0max = y0.max() * expt
    y0min = y0.min() * expt
    y1max = y1.max() * expt
    y1min = y1.min() * expt

    plt.figure()

    plt.subplot(211)
    plt.plot(t, y0)
    plt.plot(t, y0f)
    plt.plot(t, y0min)
    plt.plot(t, y0max)

    plt.subplot(212)
    plt.plot(t, y1)
    plt.plot(t, y0f)
    plt.plot(t, y1min)
    plt.plot(t, y1max)


def alpharc_plots():
    def plot_p(p, ax=None):
        if ax is None:
            ax = plt.gca()

        t = dt * np.arange(int(tmax / dt))
        y0 = alpha_series(p, t)[0]
        y1 = alpharc_series(p, t)[0]

        membrane = nengo.synapses.Lowpass(tau_rc)
        y0f = membrane.filt(y0, y0=0, dt=dt)

        expt = -np.expm1(-t/tau_rc)
        y1max = y1.max() * expt
        y1min = y1.min() * expt

        ax.plot(t, y1, 'k:', label=r"$s_3$")
        ax.plot(t, y0f, label=r"$h_m\ast s_2$")
        ax.plot(t, y1min, 'k--', label=r"$\min(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")
        ax.plot(t, y1max, 'k--', label=r"$\max(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")
        ax.set_xlabel('time [s]')
        ax.set_ylabel('membrane voltage')
        ax.set_title('p = %0.2f' % p)

    plt.figure()
    plt.subplot(121)
    plot_p(0.05)

    plt.subplot(122)
    plot_p(0.01)

    plt.legend(loc=4)

    sns.despine()
    plt.tight_layout()

    plt.savefig('alpharc_empirical.pdf')


def alpharc_distribution():
    def plot_p(p, ax=None):
        if ax is None:
            ax = plt.gca()

        t = dt * np.arange(int(tmax / dt))

        dp = 0.1
        to = p * dp * np.arange(int(1.0 / dp))
        T = t[:, None] + to[None, :]

        y0 = alpha_series(p, T)
        y1 = alpharc_series(p, t)

        membrane = nengo.synapses.Lowpass(tau_rc)
        y0f = membrane.filt(y0, y0=0, dt=dt, axis=0)

        expt = -np.expm1(-t/tau_rc)
        y1max = y1.max() * expt
        y1min = y1.min() * expt

        # sns.tsplot(y0f.T, ax=ax)
        sns.tsplot(y0f.T, time=t, err_style='unit_traces')

        # ax.plot(t, y1, 'k:', label=r"$s_3$")
        # ax.plot(t, y0f, label=r"$h_m\ast s_2$")
        ax.plot(t, y1min, 'k:', label=r"$\min(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")
        ax.plot(t, y1max, 'k--', label=r"$\max(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")

        ax.set_xlabel('time [s]')
        ax.set_ylabel('membrane voltage')
        ax.set_title('p = %0.2f' % p)

    plt.figure(figsize=(6, 5))
    # plt.figure(figsize=(8, 4))

    # ps = [0.05, 0.02, 0.01]
    ps = [0.1, 0.05, 0.02, 0.01]
    # ps = [0.05, 0.02, 0.01, 0.005]

    # r = 1
    # c = len(ps)
    r = 2
    c = 2

    for i, p in enumerate(ps):
        plt.subplot(r, c, i+1)
        plot_p(p)

    plt.legend(loc=4)

    sns.despine()
    plt.tight_layout()

    plt.savefig('alpharc_distribution.pdf')


def alpharc_multineuron():
    rng = np.random.RandomState(5)
    # u = rng.uniform(low=0, high=1, size=9)

    def plot_p(p, ax=None):
        if ax is None:
            ax = plt.gca()

        t = dt * np.arange(int(tmax / dt))

        y0s = []
        for _ in range(10):
            u = rng.uniform(low=0, high=1, size=5)
            # u = rng.uniform(low=0, high=1, size=9)
            to = p * u
            T = t[:, None] + to[None, :]
            y0i = alpha_series(p, T).mean(1)
            y0s.append(y0i)

        y0 = np.column_stack(y0s)
        y1 = alpharc_series(p, t)

        membrane = nengo.synapses.Lowpass(tau_rc)
        y0f = membrane.filt(y0, y0=0, dt=dt, axis=0)

        expt = -np.expm1(-t/tau_rc)
        y1max = y1.max() * expt
        y1min = y1.min() * expt

        # sns.tsplot(y0f.T, ax=ax)
        sns.tsplot(y0f.T, time=t, err_style='unit_traces')

        # ax.plot(t, y1, 'k:', label=r"$s_3$")
        # ax.plot(t, y0f, label=r"$h_m\ast s_2$")
        ax.plot(t, y1min, 'k:', label=r"$\min(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")
        ax.plot(t, y1max, 'k--', label=r"$\max(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")

        ax.set_xlabel('time [s]')
        ax.set_ylabel('membrane voltage')
        ax.set_title('p = %0.2f' % p)

    plt.figure(figsize=(6, 5))
    # plt.figure(figsize=(8, 4))

    # ps = [0.05, 0.02, 0.01]
    ps = [0.1, 0.05, 0.02, 0.01]
    # ps = [0.05, 0.02, 0.01, 0.005]

    # r = 1
    # c = len(ps)
    r = 2
    c = 2

    for i, p in enumerate(ps):
        plt.subplot(r, c, i+1)
        plot_p(p)

    plt.legend(loc=4)

    sns.despine()
    plt.tight_layout()

    plt.savefig('alpharc_multineuron.pdf')


def alpharc_multiperiod():
    rng = np.random.RandomState(5)
    ninputs = 9
    ntrials = 100

    def plot_ps(p0, p1, ax=None):
        if ax is None:
            ax = plt.gca()

        ps = np.linspace(p0, p1, ninputs)

        t = dt * np.arange(int(tmax / dt))
        u = rng.uniform(low=0, high=1, size=(ntrials, len(ps)))

        y0s = []
        y1s = []
        for i in range(ntrials):
            to = ps * u[i]
            T = t[:, None] + to[None, :]
            y0i = alpha_series(ps[None, :], T).mean(1)
            y0s.append(y0i)

            y1i = alpharc_series(ps, to).mean()
            y1s.append(y1i)

        y0 = np.column_stack(y0s)
        # y1 = alpharc_series(ps[None, :], t[:, None]).mean(1)
        y1 = np.array(y1s)

        membrane = nengo.synapses.Lowpass(tau_rc)
        y0f = membrane.filt(y0, y0=0, dt=dt, axis=0)

        expt = -np.expm1(-t/tau_rc)
        y1max = y1.max() * expt
        y1min = y1.min() * expt

        # sns.tsplot(y0f.T, ax=ax)
        sns.tsplot(y0f.T, time=t, err_style='unit_traces')

        # ax.plot(t, y1, 'k:', label=r"$s_3$")
        # ax.plot(t, y0f, label=r"$h_m\ast s_2$")
        ax.plot(t, y1min, 'r:', label=r"$\min(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")
        ax.plot(t, y1max, 'r--', label=r"$\max(s_3) (1 - e^{-t/\tau_\mathrm{RC}})$")

        ax.set_xlabel('time [s]')
        ax.set_ylabel('membrane voltage')
        ax.set_title('p $\in$ [%0.2f, %0.2f]' % (ps.min(), ps.max()))

    plt.figure(figsize=(6, 5))
    # plt.figure(figsize=(8, 4))

    p01s = [(0.1, 0.2),
            (0.05, 0.1),
            (0.02, 0.05),
            (0.01, 0.02)]
    r = 2
    c = 2

    for i, p01 in enumerate(p01s):
        plt.subplot(r, c, i+1)
        plot_ps(*p01)

    plt.legend(loc=4)

    sns.despine()
    plt.tight_layout()

    plt.savefig('alpharc_multiperiod.pdf')


if __name__ == '__main__':
    # alpharc_plots()
    alpharc_distribution()
    alpharc_multineuron()
    alpharc_multiperiod()

    plt.show()

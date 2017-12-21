import inspect

import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms

import seaborn as sns
sns.set_style('white')


def call_function(f, data, **kwargs):
    argspec = inspect.getargspec(f)
    defaults = () if argspec.defaults is None else argspec.defaults
    args = argspec.args[:len(argspec.args) - len(defaults)]
    return f(*[data[a] for a in args], **kwargs)


def print_test_errors(Ytest, learners):
    Ytestrms = rms(Ytest, axis=1).mean()
    for learner in learners:
        e = rms(learner['Ytest'] - Ytest, axis=1).mean() / Ytestrms
        print("%s: %0.3e" % (learner['name'], e))


def output_plot(dt, t, ystar, learners, tmin=None, tmax=None, ax=None):
    ax = plt.gca() if ax is None else ax
    ys = [learner['y'] for learner in learners]

    if tmin is not None or tmax is not None:
        tmin = t[0] if tmin is None else tmin
        tmax = t[-1] if tmax is None else tmax
        tmask = (t >= tmin) & (t <= tmax)
        t = t[tmask]
        ystar = ystar[tmask]
        ys = [y[tmask] for y in ys]

    # dinds = slice(0, 2)
    dinds = list(range(2))
    dstyles = ['-', ':']
    # dstyles = ['-', '-.']

    vsynapse = Alpha(0.01, default_dt=dt)
    ystar = ystar[:, dinds]
    ys = [vsynapse.filtfilt(y[:, dinds]) for y in ys]
    for k, (dind, dstyle) in enumerate(zip(dinds, dstyles)):
        ax.set_color_cycle(None)
        ax.plot(t, ystar[:, k], 'k', linestyle=dstyle)
        for y in ys:
            ax.plot(t, y[:, k], linestyle=dstyle)

    plt.ylim((-1.5, 1.5))


def output1_plot(dt, t, ystar, learner, tmin=None, tmax=None, ax=None):
    ax = plt.gca() if ax is None else ax
    y = learner['y']

    if tmin is not None or tmax is not None:
        tmin = t[0] if tmin is None else tmin
        tmax = t[-1] if tmax is None else tmax
        tmask = (t >= tmin) & (t <= tmax)
        t = t[tmask]
        ystar = ystar[tmask]
        y = y[tmask]

    # dinds = slice(0, 2)
    dinds = list(range(2))

    vsynapse = Alpha(0.01, default_dt=dt)
    ystar = ystar[:, dinds]
    # y = y[:, dinds]
    y = vsynapse.filtfilt(y[:, dinds])

    ax.plot(t, y[:, dinds])
    ax.set_color_cycle(None)
    ax.plot(t, ystar[:, dinds], ':')

    plt.legend(['dim %d' % (i+1) for i in range(len(dinds))], loc='best')

    plt.xlim((tmin, tmax))
    plt.ylim((-1.5, 1.5))

    # plt.xticks((tmin, tmax))
    # ax.set_xticks(ax.get_xticks()[::2])
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().get_major_formatter().set_scientific(False)


def spike_plot(dt, t, y, tmin=None, tmax=None, ax=None):
    from nengo.utils.matplotlib import rasterplot
    ax = plt.gca() if ax is None else ax

    # y = learner['hs']
    if tmin is not None or tmax is not None:
        tmin = t[0] if tmin is None else tmin
        tmax = t[-1] if tmax is None else tmax
        tmask = (t >= tmin) & (t <= tmax)
        t = t[tmask]
        y = y[tmask]

    rasterplot(t, y)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().get_major_formatter().set_scientific(False)


def error_plot(dt, prestime, t, ystar, learners, tmin=None, tmax=None, ax=None):
    ax = plt.gca() if ax is None else ax
    es = [learner['e'] for learner in learners]

    if tmin is not None or tmax is not None:
        tmin = t[0] if tmin is None else tmin
        tmax = t[-1] if tmax is None else tmax
        tmask = (t >= tmin) & (t <= tmax)
        t = t[tmask]
        es = [e[tmask] for e in es]

    vsynapse = Alpha(0.01, default_dt=dt)

    # esynapse = Alpha(1*prestime, default_dt=dt)
    # esynapse = Alpha(1*prestime, default_dt=dt)
    esynapse = Alpha(5*prestime, default_dt=dt)
    # esynapse = Alpha(20*prestime, default_dt=dt)

    # yrms = rms(ystar, axis=1).mean()
    yrms = rms(vsynapse.filtfilt(ystar), axis=1).mean()
    # yrms = rms(esynapse.filtfilt(ystar), axis=1).mean()
    # print(yrms)
    for e in es:
        # erms = esynapse.filtfilt(rms(e, axis=1) / yrms)
        # erms = rms(esynapse.filtfilt(e), axis=1) / yrms
        # erms = rms(vsynapse.filtfilt(e), axis=1) / yrms
        erms = esynapse.filtfilt(rms(vsynapse.filtfilt(e), axis=1) / yrms)
        plt.plot(t, erms)


def cosyne_plot(dt, prestime, t, ystar, learners, n_test_pre, n_train,
                offline_data=None):
    vsynapse = Alpha(0.01, default_dt=dt)

    learner = learners[-1]

    n_show_pre = 10
    n_show_post = 10
    assert n_show_pre <= n_test_pre

    t0 = n_test_pre*prestime
    t1 = t0 + n_train*prestime
    # t2 = t1 + n_test_post*prestime

    tpre0 = 0
    tpre1 = tpre0 + n_show_pre*prestime
    tpost0 = t1
    # tpost0 = t1 + 10*prestime
    tpost1 = tpost0 + n_show_post*prestime

    plt.figure(figsize=(6.4, 7))

    # subplot_shape = (2, 3)
    subplot_shape = (3, 3)

    ax = plt.subplot2grid(subplot_shape, (0, 2))
    # output_plot(dt, t, ystar, learners, tmin=tpre0, tmax=tpre1, ax=ax)
    output1_plot(dt, t, ystar, learner, tmin=tpre0, tmax=tpre1, ax=ax)
    plt.title('Pre-learning output')

    ax = plt.subplot2grid(subplot_shape, (1, 2))
    # output_plot(dt, t, ystar, learners, tmin=tpost0, tmax=tpost1, ax=ax)
    output1_plot(dt, t, ystar, learner, tmin=tpost0, tmax=tpost1, ax=ax)
    plt.xlabel('simulation time [s]')
    plt.title('Post-learning output')

    hid_spikes0 = learner['hs'][0][:, :40]
    hid_spikes1 = learner['hs'][-1][:, :40]

    ax = plt.subplot2grid(subplot_shape, (0, 0))
    spike_plot(dt, t, hid_spikes0, tmin=tpre0, tmax=tpre1, ax=ax)
    plt.title('Pre-learning spikes 1')

    ax = plt.subplot2grid(subplot_shape, (1, 0))
    spike_plot(dt, t, hid_spikes0, tmin=tpost0, tmax=tpost1, ax=ax)
    plt.xlabel('simulation time [s]')
    plt.title('Post-learning spikes 1')

    ax = plt.subplot2grid(subplot_shape, (0, 1))
    spike_plot(dt, t, hid_spikes1, tmin=tpre0, tmax=tpre1, ax=ax)
    plt.title('Pre-learning spikes 2')

    ax = plt.subplot2grid(subplot_shape, (1, 1))
    spike_plot(dt, t, hid_spikes1, tmin=tpost0, tmax=tpost1, ax=ax)
    plt.xlabel('simulation time [s]')
    plt.title('Post-learning spikes 2')

    ax = plt.subplot2grid(subplot_shape, (2, 0), colspan=3)
    error_plot(dt, prestime, t, ystar, [learner], tmin=t0, tmax=t1, ax=ax)
    if offline_data:
        Yrms = rms(offline_data['Y'], axis=1).mean()
        eo = offline_data['learners'][-1]['batch_errors'] / Yrms
        dto = prestime*offline_data['n_per_batch']
        to = t0 + dto*np.arange(len(eo))

        tmask = (to >= t0) & (to <= t1)
        to = to[tmask]
        eo = eo[tmask]

        esynapse = Alpha(5*prestime, default_dt=dto)
        eo = esynapse.filtfilt(eo)

        plt.plot(to, eo, 'k:')
        plt.xlim((t0, t1))
        plt.legend(['spiking', 'non-spiking'], loc='best')

    plt.xlabel('simulation time [s]')
    plt.ylabel('normalized RMS error')
    plt.title('Error')

    plt.tight_layout()


def trials_error_plot(dt, prestime, t, ystar, learners, n_test_pre, n_train):
    vsynapse = Alpha(0.01, default_dt=dt)

    t1 = (n_test_pre + n_train)*prestime
    t2 = t1 + 20*prestime

    plt.figure()

    ax = plt.subplot(211)
    # plot test output
    output_plot(dt, t, ystar, learners, tmin=t1, tmax=t2, ax=ax)

    ax = plt.subplot(212)
    esynapse = Alpha(5*prestime, default_dt=dt)
    # yrms = rms(esynapse.filtfilt(ystar), axis=1).mean()
    # for learner in learners:
    #     e = rms(esynapse.filtfilt(learner['e']), axis=1) / yrms
    #     plt.plot(t, e)

    # yrms = esynapse.filtfilt(rms(ystar, axis=1)).mean()
    # for learner in learners:
    #     e = esynapse.filtfilt(rms(learner['e'], axis=1)) / yrms
    #     plt.plot(t, e)

    plt.xlabel('training time [s]')
    plt.ylabel('normalized RMS training error')
    plt.tight_layout()


def error_layers_plots(dt, t, learners):
    vsynapse = Alpha(0.01, default_dt=dt)

    for learner in [l for l in learners if 'els' in l]:
        plt.figure()
        plt.subplot(211)
        dind = 0

        e = vsynapse.filtfilt(learner['e'])
        els = [vsynapse.filtfilt(el) for el in learner['els']]
        plt.plot(t, e[:, dind])
        [plt.plot(t, el[:, dind]) for el in els]

        plt.subplot(212)
        plt.plot(t, norm(e, axis=1))
        [plt.plot(t, norm(el, axis=1)) for el in els]

    plt.show()


def show_all_plots(data, offline_data=None):
    call_function(print_test_errors, data)
    call_function(cosyne_plot, data, offline_data=offline_data)
    # call_function(trials_error_plot, data)
    # call_function(error_layers_plots, data)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('loadfile', help="Result to load")
    parser.add_argument('offlinefile', nargs='?', default=None, help="Offline result")
    args = parser.parse_args()

    data = np.load(args.loadfile)
    offline_data = np.load(args.offlinefile) if args.offlinefile else None

    show_all_plots(data, offline_data)

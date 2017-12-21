import inspect

import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms
from nengo_extras.data import load_mnist


def call_function(f, data, **kwargs):
    argspec = inspect.getargspec(f)
    defaults = () if argspec.defaults is None else argspec.defaults
    args = argspec.args[:len(argspec.args) - len(defaults)]
    return f(*[data[a] for a in args], **kwargs)


def print_test_errors(learners, mnist=None):
    _, (_, Ytest) = mnist
    for learner in learners:
        e = (np.argmax(learner['Ttest'], axis=1) != Ytest).mean()
        print("%s: %0.2e" % (learner['name'], 100 * e))


def trials_error_plot(prestime, t, ystar, learners):
    pdt = 0.01
    vsynapse = Alpha(0.02, default_dt=pdt)

    plt.figure()
    dinds = slice(0, 2)

    plt.subplot(211)
    plt.plot(t, ystar[:, dinds])
    for learner in learners:
        y = vsynapse.filtfilt(learner['y'][:, dinds])
        plt.plot(t, y)
    plt.ylabel('outputs')

    plt.subplot(212)
    esynapse = Alpha(5*prestime, default_dt=pdt)
    for learner in learners:
        e = norm(esynapse.filtfilt(learner['e']), axis=1)
        plt.plot(t, e)
    plt.ylabel('errors')


# def error_layers_plots(dt, t, learners):
#     vsynapse = Alpha(0.01, default_dt=dt)

#     for learner in [l for l in learners if 'els' in l]:
#         plt.figure()
#         plt.subplot(211)
#         dind = 0

#         e = vsynapse.filtfilt(learner['e'])
#         els = [vsynapse.filtfilt(el) for el in learner['els']]
#         plt.plot(t, e[:, dind])
#         [plt.plot(t, el[:, dind]) for el in els]

#         plt.subplot(212)
#         plt.plot(t, norm(e, axis=1))
#         [plt.plot(t, norm(el, axis=1)) for el in els]

#     plt.show()


def show_all_plots(data, mnist):
    call_function(print_test_errors, data, mnist=mnist)
    call_function(trials_error_plot, data)
    plt.savefig('online_trials_error_plot.pdf')
    # call_function(error_layers_plots, data)
    # plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('loadfile', help="Result to load")
    args = parser.parse_args()

    mnist = load_mnist('~/data/mnist.pkl.gz')
    data = np.load(args.loadfile)

    show_all_plots(data, mnist)

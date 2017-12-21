import inspect

import matplotlib.pyplot as plt
import numpy as np

from nengo.synapses import Alpha
from nengo.utils.numpy import rms

import seaborn as sns

sns.set_style('white')


def call_function(f, data):
    args = inspect.getargspec(f).args
    return f(*[data[a] for a in args])


def cosyne_plot(learners, n_per_batch, Y):
    Yrms = rms(Y, axis=1).mean()

    plt.figure(figsize=(4, 3))

    def plot_batches(x, label=None, color=None):
        filt = Alpha(10, default_dt=n_per_batch)
        y = filt.filtfilt(x) / Yrms
        batch_inds = n_per_batch * np.arange(len(x))
        plt.plot(batch_inds, y, label=label, color=color)
        # plt.semilogy(batch_inds, y, label=label, color=color)

    for learner in learners:
        plot_batches(learner['batch_errors'], label=learner['name'])

    # plt.ylim((0.1, 3))

    plt.xlabel('# examples')
    plt.ylabel('normalized RMS error')
    # plt.legend(loc=1)
    plt.legend(loc='best')

    plt.tight_layout()


def show_all_plots(data):
    call_function(cosyne_plot, data)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('loadfile', help="Result to load")
    args = parser.parse_args()

    data = np.load(args.loadfile)

    show_all_plots(data)

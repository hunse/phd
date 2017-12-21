import os

import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.synapses import Alpha
from nengo.utils.numpy import norm, rms


def load_trials_from_files(filedir):
    files = os.listdir(filedir)
    files = [f for f in files if f.startswith('trial_')]

    trials = []
    for filename in files:
        data = np.load(os.path.join(filedir, filename))
        args = dict(eta=data['eta'], prestime=data['prestime'])
        result = dict(loss=data['cost'])
        trials.append(dict(args=args, result=result))

    return trials


def print_best_trial(trials):
    best_trial = trials[np.argmin([trial['result']['loss'] for trial in trials])]
    print(best_trial)


def plot_loss(trials):
    eta = np.array([trial['args']['eta'] for trial in trials])
    prestime = np.array([trial['args']['prestime'] for trial in trials])
    loss = np.array([trial['result']['loss'] for trial in trials])
    logeta = np.log10(eta)
    logprestime = np.log10(prestime)
    logloss = np.log10(loss)

    # etamin, etamax = 3.2e-3, 3.2
    etamin, etamax = 3.2e-3, 0.9
    m = (eta >= etamin) & (eta <= etamax)
    eta, prestime, loss = eta[m], prestime[m], loss[m]
    logeta, logprestime, logloss = logeta[m], logprestime[m], logloss[m]

    ax = plt.gca()
    contours = ax.tricontourf(logeta, logprestime, logloss)
    cbar = plt.colorbar(contours)

    # ax.set_xlim(np.log10(etamin), np.log10(etamax))
    ax.set_xticklabels(['%0.2g' % 10**x for x in ax.get_xticks()])
    ax.set_yticklabels(['%0.2g' % 10**x for x in ax.get_yticks()])
    cbar.set_ticklabels(['%0.2g' % 10**x for x in cbar.locator()])
    ax.set_xlabel('eta [s$^{-1}$]')
    ax.set_ylabel('presentation time [s]')
    cbar.set_label('relative RMS test error')



# def show_all_plots(trials):



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filedir', help="Directory to load from")
    args = parser.parse_args()

    trials = load_trials_from_files(args.filedir)

    print_best_trial(trials)

    plot_loss(trials)
    plt.show()

import datetime
import fnmatch
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

import nengo
import nengo.utils.numpy as npext
from nengo_extras.data import load_mnist, one_hot_from_labels
# from nengo_extras.vision import Gabor, Mask, ciw_encoders
from nengo_extras.vision import Gabor, Mask, ciw_encoders, cd_encoders_biases

import hunse_thesis.solvers
from hunse_thesis.vision import (
    percentile_encoders_intercepts, scale_encoders_intercepts)

sns.set_style('white')
# sns.set(context='paper', style='ticks')
sns.set(context='paper', style='ticks', palette='deep')
# sns.set(context='paper', style='ticks', palette='dark')

results_dir = 'mnist_compare'
s_in = (28, 28)
n_in = np.prod(s_in)
n_out = 10

GPU_DEFAULT = False


def scale_xaxis_numerically(axis, points, log=False):
    points = np.asarray(points)

    n = len(points)
    if log:
        p0, p1 = np.log(points[0]), np.log(points[-1])
        p = np.log(points)
    else:
        p0, p1 = points[0], points[-1]
        p = points

    mapping = (p - p0) * ((n - 1) / (p1 - p0))

    for line in axis.get_lines():
        line.set_xdata([mapping[int(x)] for x in line.get_xdata()])

    for pathc in axis.collections:
        X = pathc.get_offsets()
        X[:, 0] = mapping
        pathc.set_offsets(X)

    axis.set_xticks(mapping)


def get_loadpattern(enc_method, dec_method, n_hid, spiking=False, gpu=None):
    spiking = 'spiking_' if spiking else '*'
    gpu = '*' if gpu is None else 'gpu_' if gpu else 'cpu_'
    return '%s_%s_%d_%s%s*.npz' % (enc_method, dec_method, n_hid, spiking, gpu)


def get_savepath(enc_method, dec_method, n_hid, spiking=False, gpu=None):
    spiking = 'spiking_' if spiking else ''
    gpu = '' if gpu is None else 'gpu_' if gpu else 'cpu_'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = '%s_%s_%d_%s%s%s.npz' % (
        enc_method, dec_method, n_hid, spiking, gpu, timestamp)
    return os.path.join(results_dir, filename)


def get_method_files(enc_method, dec_method, n_hid, spiking=False, gpu=None):
    pattern = get_loadpattern(enc_method, dec_method, n_hid, spiking=spiking, gpu=gpu)
    filenames = fnmatch.filter(os.listdir(results_dir), pattern)
    return [os.path.join(results_dir, filename) for filename in filenames]


def get_encoders(full_method, n_hid, rng=None):
    match = re.match('(.*?)\-([0-9]+)', full_method)
    if match:
        method, s = match.groups()
        rf_shape = (int(s), int(s))
    else:
        method = full_method
        rf_shape = None

    neuron_type = nengo.LIF()
    max_rates = 100 * np.ones(n_hid)
    # max_rates = 200 * np.ones(n_hid)

    # intercepts = rng.uniform(0.01, 0.3, size=n_hid)
    intercepts = 0.1 * np.ones(n_hid)
    # intercepts = 0.0 * np.ones(n_hid)
    # intercepts = -0.5 * np.ones(n_hid)
    gain, bias = neuron_type.gain_bias(max_rates, intercepts)

    if method == 'full':
        encoders = rng.normal(size=(n_hid,) + s_in).reshape((n_hid, n_in))
        encoders /= npext.norm(encoders, axis=1, keepdims=True)
    elif method == 'mask':
        encoders = Mask(s_in).populate(
            rng.normal(size=(n_hid,) + rf_shape), rng=rng, flatten=True)
        encoders /= npext.norm(encoders, axis=1, keepdims=True)
    elif method == 'gabor':
        from hunse_thesis.dists import LogUniform
        gabor = Gabor()
        # gabor = Gabor(freq=LogUniform(np.log(0.4), np.log(2.)))
        encoders = Mask(s_in).populate(
            gabor.generate(n_hid, rf_shape, rng=rng), rng=rng, flatten=True)
        # encoders, intercepts = percentile_encoders_intercepts(
        #     train_images, encoders, percentile=50)
        # encoders, intercepts = scale_encoders_intercepts(
        #     encoders, intercepts, trainX=train_images, norm='std')
    elif method == 'gaborstat':
        from nengo_extras.vision import gabors_for_images
        gabors = gabors_for_images(
            train_images.reshape(-1, 28, 28), n_hid, rf_shape, rng=rng)
        encoders = Mask(s_in).populate(gabors, rng=rng, flatten=True)
    elif method == 'pca':
        from pca_encoders import generate_pca_encoders
        encoders = generate_pca_encoders(n_hid, train_images, rng=rng)
        encoders /= npext.norm(encoders, axis=1, keepdims=True)
    elif method == 'sample':
        encoders = train_images[rng.randint(train_images.shape[0], size=n_hid)]
        if rf_shape is not None:
            encoders *= Mask(s_in).generate(
                n_hid, rf_shape, rng=rng, flatten=True)
        encoders[encoders < 0] = 0
        encoders /= npext.norm(encoders, axis=1, keepdims=True)
        # A = np.dot(train_images[:1000], encoders.T)
        # plt.subplot(211)
        # plt.hist(encoders.ravel(), bins=30)
        # plt.subplot(212)
        # plt.hist(A.ravel(), bins=30)
        # plt.show()
        # assert 0
    elif method == 'ciw':
        encoders = ciw_encoders(n_hid, train_images, train_labels, rng=rng)
        encoders, intercepts = percentile_encoders_intercepts(
            train_images, encoders, percentile=50)
    elif method == 'ciw-rf':
        encoders = ciw_encoders(n_hid, train_images, train_labels, rng=rng)
        encoders *= Mask(s_in).generate(
            n_hid, rf_shape, rng=rng, flatten=True)
        encoders /= npext.norm(encoders, axis=1, keepdims=True)
    elif method == 'cd':
        encoders, biases = cd_encoders_biases(
            n_hid, train_images, train_labels, rng=rng)
        intercepts = -biases
        norms = npext.norm(encoders, axis=1)
        encoders /= norms[:, None]
        intercepts /= norms
    elif method == 'cd-rf':
        mask = Mask(s_in).generate(n_hid, rf_shape, rng=rng, flatten=True)
        encoders, biases = cd_encoders_biases(
            n_hid, train_images, train_labels, rng=rng, mask=mask)
        intercepts = -biases
        norms = npext.norm(encoders, axis=1)
        encoders /= norms[:, None]
        intercepts /= norms
    elif method == 'dist':
        classes = np.unique(train_labels)
        assert n_hid % len(classes) == 0
        n_hid_per_class = n_hid / len(classes)

        encoders = []
        for label in classes:
            X = train_images[train_labels == label]

            mu = X.mean(axis=0, keepdims=True)
            Xc = X - mu
            C = np.dot(Xc.T, Xc) / (Xc.shape[0]) + 1e-5 * np.eye(Xc.shape[1])
            L = np.linalg.cholesky(C)

            samples = np.dot(rng.normal(size=(n_hid_per_class, L.shape[0])), L) + mu
            # samples = np.dot(rng.uniform(-1, 1, size=(n_hid_per_class, L.shape[0])), L) + mu

            encoders.append(samples)

        encoders = np.vstack(encoders)
        encoders *= Mask(s_in).generate(
            encoders.shape[0], rf_shape, rng=rng, flatten=True)
    else:
        raise ValueError(method)

    bias -= gain * intercepts
    return neuron_type, encoders, gain, bias


def get_solver(full_method):
    match = re.match('(.+?)\-(.+)', full_method)
    if match:
        method, s = match.groups()
        reg = float(s)
    else:
        method = full_method

    if method == 'LstsqL2':
        return nengo.solvers.LstsqL2(reg=reg)
    elif method == 'LstsqClassifier':
        return hunse_thesis.solvers.LstsqClassifier(reg=reg)
    elif method == 'LstsqClassifier2':
        return hunse_thesis.solvers.LstsqClassifier(reg=reg, weight_power=2)
    elif method == 'LstsqClassifierParts':
        return hunse_thesis.solvers.LstsqClassifierParts(reg=reg)
    elif method == 'Softmax':
        return hunse_thesis.solvers.Softmax(reg=reg)
    elif method == 'Hinge':
        return hunse_thesis.solvers.HingeLoss(reg=reg)
    elif method == 'LDA':
        return hunse_thesis.solvers.LDA_OVA()
    elif method == 'LstsqLOO':
        return hunse_thesis.solvers.LstsqLOO()
    elif method == 'LstsqClassifierLOO':
        return hunse_thesis.solvers.LstsqClassifierLOO()
    else:
        raise ValueError(method)

    return solver


def load_data():
    global train_images, train_labels, train_targets
    global test_images, test_labels, test_targets

    train, test = load_mnist('~/data/mnist.pkl.gz')
    # train = (train[0][:10000], train[1][:10000])

    train_images, train_labels = train
    test_images, test_labels = test
    for images in [train_images, test_images]:
        images[:] = 2 * images - 1  # normalize to -1 to 1

    train_targets = one_hot_from_labels(train_labels, classes=10)
    test_targets = one_hot_from_labels(test_labels, classes=10)

    assert train_images.shape[1] == n_in
    assert train_targets.shape[1] == n_out


def load_trial(filepath, default=None):
    data = np.load(filepath)
    return data['train_error'], data['test_error'], (
        data['spiking_error'] if 'spiking_error' in data else default), (
        data['train_time'] if 'train_time' in data else default)


def load_trial_values(filepath, *values, **kwargs):
    default = kwargs.pop('default', None)
    data = np.load(filepath)
    asitem = lambda a: a.item() if a.shape == () else a
    return tuple(asitem(data[value]) if value in data else default
                 for value in values)


def print_method_files(enc_method, dec_method, n_hid, spiking=False):
    files = get_method_files(enc_method, dec_method, n_hid, spiking=spiking)
    for filename in files:
        train_error, test_error, spiking_error, train_time = load_trial(
            filename, default=np.nan)
        act_info, = load_trial_values(filename, 'test_act_info')
        print("%s: %0.2f, %0.2f, %0.2f, %0.2f" % (
            filename, train_error, test_error, spiking_error, train_time))
        print("    Test acts: %0.2f, %0.2f%%, %0.2f" % tuple(
            act_info[x] for x in ['mean', 'active', 'active_mean']))


def run_trial(enc_method, dec_method, n_hid, save=True, seed=None,
              spiking=False, gpu=None):
    if 'train_images' not in locals():
        load_data()

    if seed is None:
        seed = np.random.randint(2**31-1)
    rng = np.random.RandomState(seed)

    neuron_type, encoders, gain, bias = get_encoders(enc_method, n_hid, rng=rng)
    solver = get_solver(dec_method)

    if 1:
        # scale encoders based on local responses
        A = np.dot(train_images[:1000], encoders.T)
        r = 2 * A.std(axis=0)
        encoders /= np.maximum(1., r)[:, None]
        # print((r < 1.).mean())

    ens_params = dict(
        eval_points=train_images,
        neuron_type=neuron_type,
        encoders=encoders,
        normalize_encoders=False,
        gain=gain,
        bias=bias,
        )

    # n_presentations = 100
    # n_presentations = 1000
    n_presentations = 10000

    # tau_s = 0.005
    # synapse = nengo.synapses.Alpha(tau_s)

    # presentation_time = 0.05
    # presentation_time = 0.06
    presentation_time = 0.07
    # presentation_time = 0.08
    # presentation_time = 0.1

    # clip_time = tau_s
    clip_time = 0.005


    with nengo.Network(seed=3) as model:
        u = nengo.Node(nengo.processes.PresentInput(test_images, presentation_time))
        a = nengo.Ensemble(n_hid, n_in, **ens_params)
        v = nengo.Node(size_in=n_out)
        nengo.Connection(u, a, synapse=None)
        conn = nengo.Connection(
            a, v, synapse=None,
            eval_points=train_images, function=train_targets, solver=solver)
        vp = nengo.Probe(v)


    if gpu is None:
        gpu = GPU_DEFAULT

    if gpu:
        import nengo_ocl
        import hunse_thesis.solvers_ocl
        Simulator = nengo_ocl.Simulator
    else:
        Simulator = nengo.Simulator

    with Simulator(model, progress_bar=False) as sim:
        def test_model(images, labels):
            _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)
            act_info = dict(mean=acts.mean(),
                            active=100*(acts > 0).mean(),
                            active_mean=acts[acts > 0].mean())
            outs = np.dot(acts, sim.data[conn].weights.T)
            error = 100*(np.argmax(outs, axis=1) != labels).mean()
            return error, act_info

        train_error, train_act_info = test_model(train_images, train_labels)
        test_error, test_act_info = test_model(test_images, test_labels)

        spiking_error = None
        if spiking:
            sim.run(n_presentations * presentation_time)
            nt = int(round(presentation_time / sim.dt))
            ct = int(round(clip_time / sim.dt))
            blocks = sim.data[vp].reshape(n_presentations, nt, 10)

            # for start in range(0, nt, 10):
            #     errors = []
            #     for end in range(start, nt+1, 10):
            #         choices = np.argmax(blocks[:, start:end, :].mean(axis=1), axis=1)
            #         error = 100 * (choices != test_labels[:n_presentations]).mean()
            #         errors.append(error)
            #     print("Start %d: %s" % (start, ', '.join(
            #         ['%0.2f' % e for e in errors])))

            choices = np.argmax(blocks[:, ct:, :].mean(axis=1), axis=1)
            spiking_error = 100 * (choices != test_labels[:n_presentations]).mean()

    train_time = sim.data[conn].solver_info['time']
    # print("Trained in %0.2f s" % train_time)
    print("Train acts: %0.2f, %0.2f%%, %0.2f" % tuple(
        train_act_info[x] for x in ['mean', 'active', 'active_mean']))

    data = dict(
        train_error=train_error, test_error=test_error,
        spiking_error=spiking_error, train_time=train_time,
        train_act_info=train_act_info, test_act_info=test_act_info,
        seed=seed)

    # --- save
    if save:
        savepath = get_savepath(enc_method, dec_method, n_hid, spiking=spiking, gpu=gpu)
        np.savez(savepath, **data)
        print("Saved %r" % savepath)

    return data


def plot_trial(enc_method, dec_method, n_hid, spiking=False, seed=None):
    trial = run_trial(
        enc_method, dec_method, n_hid, spiking=spiking, save=False, seed=seed)
    spiking_error = trial['spiking_error']
    print("%s, %s, %s: %0.2f, %0.2f, %0.2f (%0.2f s)" % (
        enc_method, dec_method, n_hid,
        trial['train_error'], trial['test_error'],
        spiking_error if spiking_error is not None else np.nan,
        trial['train_time']))


def plot_encoder_compare():
    # Factors: method, n_hid, train/test
    enc_methods = ['full', 'mask-5', 'mask-7', 'mask-9', 'mask-11', 'mask-13', 'mask-15', 'mask-17',
                   'gabor-9', 'gabor-11', 'gabor-13', 'gabor-15', 'gabor-17',
                   'ciw', 'ciw-rf-7', 'ciw-rf-9', 'ciw-rf-11', 'ciw-rf-13', 'ciw-rf-15', 'ciw-rf-17',
                   'cd', 'cd-rf-9', 'cd-rf-11', 'cd-rf-13', 'cd-rf-15', 'cd-rf-17']

    n_hids = [100, 200, 500, 1000, 2000, 5000]
    n_trials = 10

    dec_method = 'LstsqClassifier-0.02'

    for enc_method in enc_methods:
        for n_hid in n_hids:
            files = get_method_files(enc_method, dec_method, n_hid)
            for _ in range(n_trials - len(files)):
                run_trial(enc_method, dec_method, n_hid)

    data = []
    for enc_method in enc_methods:
        for n_hid in n_hids:
            files = get_method_files(enc_method, dec_method, n_hid)
            files = files[:n_trials]

            for filepath in files:
                train_error, test_error, _, _ = load_trial(filepath)
                data.append((enc_method, n_hid, train_error, test_error))


    columns = ('method', 'n_hid', 'train_error', 'test_error')
    data = pandas.DataFrame(data=data, columns=columns)

    # # --- all
    # plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data,
    #                       kind='bar', legend=False, palette='muted')
    # ticks = [1, 2, 5, 10, 20]
    # ax = plot.fig.get_axes()[0]
    # ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    # ax.set(yticks=ticks, yticklabels=ticks)
    # ax.set_xlabel('# of neurons')
    # ax.set_ylabel('test classification error [%]')
    # ax.legend(loc=1)
    # # plt.savefig('mnist_compare_encoders_all.pdf')
    # plt.show()
    # return

    # --- masks
    data_masks = data[[m.startswith(('mask', 'full')) for m in data['method']]]
    plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data_masks,
                          kind='bar', legend=False, palette='muted')
    ticks = [1, 2, 5, 10, 20]
    ax = plot.fig.get_axes()[0]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('test classification error [%]')
    ax.legend(loc=1)
    plt.savefig('mnist_compare_encoders_masks.pdf')

    # --- gabors
    data_gabors = data[[m.startswith(('gabor', 'full')) for m in data['method']]]
    plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data_gabors,
                          kind='bar', legend=False, palette='muted')
    ticks = [1, 2, 5, 10, 20]
    ax = plot.fig.get_axes()[0]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('test classification error [%]')
    ax.legend(loc=1)
    plt.savefig('mnist_compare_encoders_gabors.pdf')

    # --- ciw
    data_ciw = data[[m.startswith(('ciw', 'full')) for m in data['method']]]
    plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data_ciw,
                          kind='bar', legend=False, palette='muted')
    ticks = [1, 2, 5, 10, 20]
    ax = plot.fig.get_axes()[0]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('test classification error [%]')
    ax.legend(loc=1)
    plt.savefig('mnist_compare_encoders_ciw.pdf')

    # --- cd
    data_cd = data[[m.startswith(('cd', 'full')) for m in data['method']]]
    plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data_cd,
                          kind='bar', legend=False, palette='muted')
    ticks = [1, 2, 5, 10, 20]
    ax = plot.fig.get_axes()[0]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('test classification error [%]')
    ax.legend(loc=1)
    plt.savefig('mnist_compare_encoders_cd.pdf')

    # --- best
    best_methods = ('full', 'mask-9', 'gabor-13', 'ciw-rf-11', 'cd-rf-11')
    data_best = data[[m in best_methods for m in data['method']]]
    plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data_best,
                          kind='bar', legend=False, palette='muted')
    ticks = [1, 2, 5, 10, 20]
    ax = plot.fig.get_axes()[0]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('test classification error [%]')
    ax.legend(loc=1)
    plt.savefig('mnist_compare_encoders_best.pdf')

    # --- best train/test
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    sns.pointplot(x='n_hid', y='train_error', hue='method', data=data_best,
                  ax=ax, linestyles=':', markers='.')
    sns.pointplot(x='n_hid', y='test_error', hue='method', data=data_best,
                  ax=ax, markers='')
    scale_xaxis_numerically(ax, n_hids, log=True)

    ticks = [0.5, 1, 2, 5, 10, 20]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)

    ax.set_xlabel('# of neurons')
    ax.set_ylabel('classification error [%]')
    ax.legend(ax.get_legend_handles_labels()[0][:len(best_methods)], best_methods, loc=1)
    # ax.legend(ax.get_legend_handles_labels()[0][-len(best_methods):], best_methods, loc=1)
    sns.despine()
    plt.tight_layout()
    plt.savefig('mnist_compare_encoders_besttt.pdf')

    # plt.show()


def plot_decoder_compare():
    # Factors: method, n_hid, train/test

    enc_method = 'gabor-13'
    # n_hids = [100, 200]
    # n_hids = [100, 200, 500, 1000, 2000]
    n_hids = [100, 200, 500, 1000, 2000, 5000]
    n_trials = 10

    dec_methods = ['LstsqL2-0.02', 'LstsqClassifier-0.02', 'Softmax-0.004', 'Hinge-0.02']
    dec_names = ['Squared 0.02', 'Weighted-Squared 0.02', 'Softmax 0.004', 'Hinge 0.02']

    for dec_method in dec_methods:
        for n_hid in n_hids:
            files = get_method_files(enc_method, dec_method, n_hid)
            for _ in range(n_trials - len(files)):
                run_trial(enc_method, dec_method, n_hid)

    data = []
    for dec_method in dec_methods:
        for n_hid in n_hids:
            files = get_method_files(enc_method, dec_method, n_hid)
            files = files[:n_trials]

            for filepath in files:
                train_error, test_error, _, _ = load_trial(filepath)
                data.append((dec_method, n_hid, train_error, test_error))


    columns = ('method', 'n_hid', 'train_error', 'test_error')
    data = pandas.DataFrame(data=data, columns=columns)

    # --- train/test error figure
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    dodge = 0
    # dodge = 0.2

    sns.pointplot(x='n_hid', y='test_error', hue='method', data=data,
                  ax=ax, markers='', dodge=dodge)
    sns.pointplot(x='n_hid', y='train_error', hue='method', data=data,
                  ax=ax, linestyles=':', markers='.', dodge=dodge)
    scale_xaxis_numerically(ax, n_hids, log=True)
    ax.set_ylim([0, 14])
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('classification error [%] (log scale)')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[-len(dec_names):], dec_names, loc=1)
    sns.despine()
    plt.savefig('mnist_compare_decoders.pdf')

    # --- test-only error figure
    plot = sns.factorplot(x='n_hid', y='test_error', hue='method', data=data,
                          kind='bar', legend=False, size=4)
    ticks = [1, 2, 5, 10, 20]
    ax = plot.fig.get_axes()[0]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    ax.legend(ax.get_legend_handles_labels()[0], dec_names, loc=1)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('test-set classification error [%]')

    plt.savefig('mnist_compare_decoders_testbar.pdf')

    # plot = sns.factorplot(x='n_hid', y='train_error', hue='method', data=data,
    #                       kind='bar', legend=False)
    # ax = plot.fig.get_axes()[0]
    # ax.legend(ax.get_legend_handles_labels()[0], dec_names, loc=1)
    # plt.savefig('mnist_compare_decoders_train.pdf')

    # plt.show()


def plot_decoder_regularization():
    # x-axis: regularization, y-axis: train and test error for each decoder type

    enc_method = 'gabor-13'
    # n_hids = [100, 200, 500, 1000, 2000, 5000]
    # n_hid = 100
    n_hid = 5000
    # n_trials = 3
    # n_trials = 10
    n_trials = 10

    # regs = np.logspace(-5, -2, 10)
    regs = np.logspace(-4, -1, 10)
    # regs = np.logspace(-3, 0, 10)
    # regs = np.logspace(-3, -1, 7)

    dec_types = ['LstsqL2', 'LstsqClassifier', 'Softmax', 'Hinge']
    dec_names = ['Squared', 'Weighted-Squared', 'Softmax', 'Hinge']

    for dec_type in dec_types:
        for reg in regs:
            dec_method = '%s-%0.2e' % (dec_type, reg)
            files = get_method_files(enc_method, dec_method, n_hid)
            for _ in range(n_trials - len(files)):
                run_trial(enc_method, dec_method, n_hid)

    data = []
    for dec_type in dec_types:
        for reg in regs:
            dec_method = '%s-%0.2e' % (dec_type, reg)

            files = get_method_files(enc_method, dec_method, n_hid)
            files = files[:n_trials]

            for filepath in files:
                train_error, test_error, _, train_time = load_trial(filepath)
                data.append((dec_type, reg, train_error, test_error, train_time))

    columns = ('type', 'reg', 'train_error', 'test_error', 'train_time')
    data = pandas.DataFrame(data=data, columns=columns)

    # --- error figure
    fig = plt.figure(figsize=(6.35, 4))
    ax = fig.add_subplot(111)
    sns.pointplot(x='reg', y='test_error', hue='type', data=data,
                  ax=ax, markers='')
    sns.pointplot(x='reg', y='train_error', hue='type', data=data,
                  ax=ax, linestyles=':', markers='.')
    ax.set_ylim([-0.1, 3])
    ax.set_xticklabels(['%0.1e' % reg for reg in regs])
    ax.set_xlabel('regularization')
    ax.set_ylabel('classification error [%]')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[-len(dec_names):], dec_names, loc=2)

    sns.despine()
    plt.tight_layout()
    plt.savefig('mnist_compare_decoder_regularization.pdf')

    # --- timing figure
    # fig = plt.figure()

    # sns.pointplot(x='reg', y='train_time', hue='type', data=data, estimator=min)
    # # sns.factorplot(x='type', y='train_time')

    # plt.show()


def plot_decoder_spiking_regularization():
    enc_method = 'gabor-13'
    # n_hids = [100, 200, 500, 1000, 2000, 5000]
    # n_hid = 100
    n_hid = 5000
    # n_trials = 1
    # n_trials = 3
    n_trials = 10

    # regs = np.logspace(-3, 0, 10)
    # regs = np.logspace(-3, 0, 10)[:9]
    regs = np.logspace(-3, 0, 10)[:7]
    # regs = np.logspace(-3, -1, 7)

    dec_types = ['LstsqL2', 'LstsqClassifier', 'Softmax', 'Hinge']
    dec_names = ['Squared', 'Weighted-Squared', 'Softmax', 'Hinge']

    for dec_type in dec_types:
        for reg in regs:
            dec_method = '%s-%0.2e' % (dec_type, reg)
            files = get_method_files(enc_method, dec_method, n_hid, spiking=True)
            for _ in range(n_trials - len(files)):
                run_trial(enc_method, dec_method, n_hid, spiking=True)

    data = []
    for dec_type in dec_types:
        for reg in regs:
            dec_method = '%s-%0.2e' % (dec_type, reg)
            files = get_method_files(enc_method, dec_method, n_hid, spiking=True)
            for filepath in files[:n_trials]:
                train_error, test_error, spiking_error, test_act_info = load_trial_values(
                    filepath, 'train_error', 'test_error', 'spiking_error', 'test_act_info')
                # ai = test_act_info.item()
                ai = test_act_info
                data.append((dec_type, reg, train_error, test_error, spiking_error,
                             ai['mean'], ai['active'], ai['active_mean']))
                # train_error, test_error, spiking_error, train_time = load_trial(filepath)
                # data.append((dec_type, reg, train_error, test_error, spiking_error, train_time))

    columns = ('type', 'reg', 'train_error', 'test_error', 'spiking_error',
               'act_mean', 'active', 'active_act_mean')
    data = pandas.DataFrame(data=data, columns=columns)

    # --- error figure
    fig = plt.figure(figsize=(6.35, 4))
    ax = fig.add_subplot(111)
    sns.pointplot(x='reg', y='spiking_error', hue='type', data=data,
                  ax=ax, markers='')
    sns.pointplot(x='reg', y='test_error', hue='type', data=data,
                  ax=ax, linestyles=':', markers='.', dodge=0.1)
    ax.set_ylim([1, 10])
    ax.set(yscale='log')
    ax.set(yticks=[1, 2, 5, 10], yticklabels=[1, 2, 5, 10])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xticklabels(['%0.1e' % reg for reg in regs])
    ax.set_xlabel('regularization')
    ax.set_ylabel('classification error [%]')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[-len(dec_names):], dec_names, loc=1)
    sns.despine()
    plt.tight_layout()

    plt.savefig('mnist_compare_decoder_spiking_regularization.pdf')

    # --- best regularization plot
    mask = np.zeros_like(data['reg'] > 0)
    for typ in dec_types:
        means = [data[(data['type'] == typ) & (data['reg'] == reg)]['spiking_error'].mean()
                 for reg in regs]
        reg = regs[np.argmin(means)]
        mask[:] = mask | ((data['type'] == typ) & (data['reg'] == reg))

    data_best = pandas.melt(
        data[mask],
        id_vars=['type', 'act_mean', 'active', 'active_act_mean'],
        value_vars=['train_error', 'test_error', 'spiking_error'],
        var_name='error_type',
        value_name='error')

    sns.factorplot(x='type', y='error', hue='error_type', data=data_best,
                   kind='bar', legend=False, size=4)
                   # kind='bar', legend=False, errcolor='0.0', size=4)

    ax = plt.gca()
    ax.set_xticklabels(dec_names, rotation=20)
    plt.legend(ax.get_legend_handles_labels()[0], ['train', 'test', 'spiking test'], loc=1)
    plt.xlabel('')
    plt.ylabel('classification error [%]')
    plt.tight_layout()

    plt.savefig('mnist_compare_decoder_spiking.pdf')

    # --- exact errors
    for typ in dec_types:
        for error in ['train_error', 'test_error', 'spiking_error']:
            mask = (data_best['type'] == typ) & (data_best['error_type'] == error)
            data_cur = data_best[mask]
            errors = data_cur['error']
            act_mean = data_cur['act_mean']
            active = data_cur['active']
            active_act_mean = data_cur['active_act_mean']
            # print(act_mean.std())
            print("%15s %15s: %0.2f (mean), %0.2f (min) "
                  "[%0.1f (act), %0.2f%% (sparse), %0.1f (active act)]" % (
                      typ, error, errors.mean(), errors.min(),
                      act_mean.mean(), active.mean(), active_act_mean.mean()))


def plot_solve_time():
    enc_method = 'gabor-13'
    n_hids = [100, 200, 500, 1000, 2000, 5000]
    # n_hids = [100, 200, 500, 1000, 2000, 5000, 10000]
    # n_hids_gpu = [100, 200, 500, 1000, 2000, 5000]
    # n_hids_cpu = [100, 200, 500, 1000]
    # n_hids_cpu = [100, 200, 500, 1000, 2000, 5000]
    n_trials = 3

    dec_methods = ['LstsqL2-0.02', 'LstsqClassifier-0.02', 'Softmax-0.004', 'Hinge-0.02']
    dec_names = ['Squared 0.02', 'Weighted-Squared 0.02', 'Softmax 0.004', 'Hinge 0.02']

    for dec_method in dec_methods:
        for gpu in [False, True]:
            for n_hid in n_hids:
            # for n_hid in (n_hids_gpu if gpu else n_hids_cpu):
                files = get_method_files(enc_method, dec_method, n_hid, gpu=gpu)
                # print(dec_method, n_hid, gpu, len(files))
                for _ in range(n_trials - len(files)):
                    run_trial(enc_method, dec_method, n_hid, gpu=gpu)

    data = []
    for dec_method in dec_methods:
        for gpu in [False, True]:
            for n_hid in n_hids:
            # for n_hid in (n_hids_gpu if gpu else n_hids_cpu):
                files = get_method_files(enc_method, dec_method, n_hid, gpu=gpu)
                for filepath in files[:n_trials]:
                    train_error, test_error, spiking_error, train_time = load_trial(filepath)
                    data.append((dec_method, n_hid, gpu, train_error, test_error, spiking_error, train_time))

    columns = ('type', 'n_hid', 'gpu', 'train_error', 'test_error', 'spiking_error', 'train_time')
    data = pandas.DataFrame(data=data, columns=columns)
    data_cpu = data[data['gpu'] == False]
    data_gpu = data[data['gpu'] == True]

    # --- error figure (for sanity)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.pointplot(x='n_hid', y='test_error', hue='type', data=data_cpu,
                  ax=ax, linestyles='-', markers='.')
    sns.pointplot(x='n_hid', y='test_error', hue='type', data=data_gpu,
                  ax=ax, linestyles=':', markers='.')
    scale_xaxis_numerically(ax, n_hids, log=True)
    ticks = [1, 2, 5, 10, 20]
    ax.set(yscale='log', ylim=(ticks[0], ticks[-1]))
    ax.set(yticks=ticks, yticklabels=ticks)
    sns.despine()
    plt.tight_layout()

    # --- timing figure
    fig = plt.figure(figsize=(6.35, 4))
    ax = fig.add_subplot(111)
    sns.pointplot(x='n_hid', y='train_time', hue='type', data=data_cpu,
                  estimator=min, ax=ax, linestyles='-', markers='.', n_boot=0)
    sns.pointplot(x='n_hid', y='train_time', hue='type', data=data_gpu,
                  estimator=min, ax=ax, linestyles=':', markers='.', n_boot=0)
    scale_xaxis_numerically(ax, n_hids, log=True)
    ax.set(yscale='log')
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('solve time [s]')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[-len(dec_names):], dec_names, loc=2)
    sns.despine()
    plt.tight_layout()
    plt.savefig('mnist_compare_solve_time.pdf')

    # plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    GPU_DEFAULT = args.gpu

    # print_method_files('gabor-13', 'LstsqL2-1.00e-03', 5000, spiking=True)
    # print_method_files('gabor-13', 'LstsqL2-1.00e-03', 5000, spiking=True)
    # print_method_files('gabor-13', 'LstsqClassifier-1.00e-02', 5000, spiking=True)

    # plot_encoder_compare()
    # plot_decoder_compare()
    # plot_decoder_regularization()
    plot_decoder_spiking_regularization()
    # plot_solve_time()

    # plot_trial('gabor-13', 'LstsqL2-0.02', 1000, seed=3)
    # plot_trial('gabor-13', 'LstsqL2-0.02', 1000, spiking=True, seed=3)

    # plot_trial('gabor-13', 'LstsqClassifier-0.02', 5000, seed=3)

    # plot_trial('gabor-13', 'Softmax-0.004', 1000, seed=3)
    # plot_trial('gabor-13', 'Hinge-0.02', 1000, seed=3)

    # plot_trial('gabor-13', 'Softmax-0.004', 5000, seed=3)
    # plot_trial('gabor-13', 'Softmax-0.004', 10000, seed=3)

    # plot_trial('gabor-13', 'LstsqL2-0.02', 1000, seed=3)
    # plot_trial('gabor-13', 'LstsqL2-0.002', 1000, seed=3)
    # plot_trial('gabor-13', 'LstsqL2-0.02', 5000, seed=3)

    # plot_trial('gabor-13', 'LstsqClassifier-0.004', 200, seed=3)
    # plot_trial('gabor-13', 'LstsqClassifier-0.02', 5000, seed=3)

    # plot_trial('gabor-13', 'LstsqLOO', 100, seed=3)
    # plot_trial('gabor-13', 'LstsqLOO', 200, seed=3)
    # plot_trial('gabor-13', 'LstsqLOO', 1000, seed=3)
    # plot_trial('gabor-13', 'LstsqLOO', 5000, seed=3)

    # plot_trial('gabor-13', 'LstsqClassifier-0.003', 100, seed=3)
    # plot_trial('gabor-13', 'LstsqClassifierLOO', 100, seed=3)
    # plot_trial('gabor-13', 'LstsqClassifierLOO', 5000, seed=3)

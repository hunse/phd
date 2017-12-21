"""
Classifier for the ImageNet ILSVRC-2012 dataset.
"""
import matplotlib.pyplot as plt
import nengo
import nengo_ocl
import nengo_extras.matplotlib as neplt
import numpy as np
from nengo.utils.matplotlib import rasterplot

from nengo_extras.data import load_ilsvrc2012
from nengo_extras.convnet import softmax
from nengo_extras.cuda_convnet import CudaConvnetNetwork, load_model_pickle


def shortest_name(names):
    names = [name.strip() for name in names if len(name.strip()) > 0]
    lens = [len(name) for name in names]
    return names[np.argmin(lens)]


rng = np.random.RandomState(9)

Xtest, Ytest, data_mean, label_names = load_ilsvrc2012(n_files=1)
Xtest = Xtest.astype('float32')

# shorten label names
label_names = [shortest_name(s.split(',')) for s in label_names]

# crop data
Xtest = Xtest[:, :, 16:-16, 16:-16]
data_mean = data_mean[:, 16:-16, 16:-16]
image_shape = Xtest.shape[1:]

# subtract mean
Xtest -= data_mean

if 0:
    # shuffle images
    i = rng.permutation(Xtest.shape[0])
    Xtest, Ytest = Xtest[i], Ytest[i]

# retrieve from https://figshare.com/s/f343c68df647e675af28
cc_model = load_model_pickle('~/data/ilsvrc2012-lif-48.pkl')


# --- Run model in Nengo
presentation_time = 0.2
c0 = 0.14  # classification start time

model = nengo.Network()
with model:
    u = nengo.Node(nengo.processes.PresentInput(Xtest, presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.003))
    nengo.Connection(u, ccnet.input, synapse=None)

    # input_p = nengo.Probe(u)
    output_p = nengo.Probe(ccnet.output)
    spike_ps = [
        nengo.Probe(ccnet.layers_by_name['conv1_neuron'].ensemble.neurons),
        nengo.Probe(ccnet.layers_by_name['conv5_neuron'].ensemble.neurons),
        nengo.Probe(ccnet.layers_by_name['fc4096b_neuron'].ensemble.neurons),
    ]
    spike_names = ['layer 1', 'layer 5', 'layer 7']

n_presentations = 5
# n_presentations = 4
# n_presentations = 2
# n_presentations = 1

with nengo_ocl.Simulator(model) as sim:
    sim.run(n_presentations * presentation_time)

t = sim.trange()
nt = int(presentation_time / sim.dt)
ct = int(c0 / sim.dt)
n_classes = ccnet.output.size_out
blocks = sim.data[output_p].reshape(n_presentations, nt, n_classes)
values = softmax(blocks[:, ct:, :].mean(axis=1), axis=1)
choices = np.argsort(values, axis=1)[:, ::-1]

spike_blocks = [
    sim.data[spike_p].reshape(n_presentations, nt, -1) for spike_p in spike_ps]


plt.figure(figsize=(6.4, 7))
rows = 2 + len(spike_ps)
cols = n_presentations

neuron_inds = [
    rng.permutation(block.shape[-1])[:40] for block in spike_blocks]

for col in range(cols):
    label0 = label_names[Ytest[col]]
    tj = sim.dt*np.arange(nt) + col*presentation_time

    # plot image
    ax = plt.subplot(rows, cols, 0*cols + col + 1)
    neplt.imshow(np.transpose(Xtest[col], (1, 2, 0)), vmin=-128, vmax=128,
                 ax=ax, axes=True)
    ax.set_title(label0)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    if col == 0:
        ax.set_ylabel('input')

    # plot spike rasters
    for i, (spike_block, neuron_ind) in enumerate(
            zip(spike_blocks, neuron_inds)):
        ax = plt.subplot(rows, cols, (i+1)*cols + col + 1)
        rasterplot(tj, spike_block[col][:, neuron_ind], ax=ax)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        if col == 0:
            ax.set_ylabel('%s neurons' % spike_names[i])

    # plot output
    ax = plt.subplot(rows, cols, (rows-1)*cols + col + 1)
    jinds = choices[col][:5]
    jlabels = [label_names[k] for k in jinds]
    jvalues = values[col][jinds]
    cells = [['%s %0.2f%%' % (label, 100*value)] for label, value in zip(
        jlabels, jvalues)]
    ax.table(cellText=cells, loc='center', fontsize=20)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    if col == 0:
        ax.set_ylabel('output')

plt.tight_layout()
plt.savefig('imagenet_demo.png')
plt.savefig('imagenet_demo.pdf')
plt.show()

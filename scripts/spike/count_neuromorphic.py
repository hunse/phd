"""
Compute efficiency statistics for my networks on neuromorphic hardware
"""
import numpy as np

dt = 0.001

flops_update = 1
flops_synop = 2

flopjoules_update = 0.25
flopjoules_synop = 0.08


class Efficiency(object):
    def __init__(self, neurons, synapses, rates, presentation_time=0.2,
                 name='', error=None):
        neurons = np.array(neurons, dtype=float)
        synapses = np.array(synapses, dtype=float)
        rates = np.array(rates, dtype=float)

        self.neurons = neurons
        self.synapses = synapses
        self.rates = rates
        self.presentation_time = presentation_time
        self.name = name
        self.error = error
        assert len(neurons) == len(rates) == len(synapses) - 1

    def flops(self):
        # --- compute flops on standard hardware
        # flops = flops_synop * synapses.sum() + flops_update * neurons.sum()
        flops0 = flops_synop * self.synapses[0]
        flops = (flops0 +
                 flops_synop * self.synapses[1:].sum() +
                 flops_update * self.neurons.sum())
        return flops

    def synops_updates(self):
        # synops = (synapses * rates).sum()
        synops = (self.synapses[1:] * self.rates).sum()
        updates = self.neurons.sum() / dt
        return synops, updates

    def energy(self):
        # --- compute energy on neuromorphic hardware
        flops0 = flops_synop * self.synapses[0]
        synops, updates = self.synops_updates()
        energy = flops0 + self.presentation_time*(
            flopjoules_synop*synops +
            flopjoules_update*updates)
        return energy

    def average_rate(self):
        return (self.rates * self.neurons).sum() / self.neurons.sum()

    def spikes_per_image(self):
        return (self.rates * self.neurons).sum() * self.presentation_time

    def print_stats(self):
        average_rate = self.average_rate()
        synops, updates = self.synops_updates()
        flops = self.flops()
        energy = self.energy()
        if self.name:
            print("%s (pt=%0.3f):" % (self.name, self.presentation_time))
        print("Average rate: %s" % (average_rate,))
        print("Synops/s = %s, updates/s = %s" % (synops, updates))
        print("flops = %0.2e, energy = %0.2e, efficiency = %0.2f" % (flops, energy, (flops / energy)))


cifar10 = Efficiency(
    name="CIFAR-10",
    error=16.46,
    neurons=[36864, 9216, 2304, 1152],
    synapses=[2764800, 14745600, 1327104, 663552, 11520],
    rates=[173.3, 99.0, 9.7, 7.2])
mnist = Efficiency(
    name="MNIST",
    error=0.88,
    neurons=[12544, 12544, 2000],
    synapses=[313600, 5017600, 6272000, 20000],
    rates=[4.6, 15.6, 4.2])
imagenet = Efficiency(
    name="ImageNet",
    error=48.2,
    neurons=[193600, 139968, 64896, 43264, 43264, 4096, 4096],
    synapses=[70276800, 223948800, 112140288, 149520384, 99680256, 37748736, 16777216, 4096000],
    rates=[178.1, 48.8, 26.6, 30.6, 35.6, 19.1, 10.7])
    # rates=[1000.0, 178.1, 48.8, 26.6, 30.6, 35.6, 19.1, 10.7],


if __name__ == '__main__':
    pts = [0.2, 0.15, 0.08, 0.06]
    for pt in pts:
        cifar10.presentation_time = pt
        cifar10.print_stats()

    pts = [0.2, 0.1, 0.06]
    for pt in pts:
        mnist.presentation_time = pt
        mnist.print_stats()

    pts = [0.2, 0.08, 0.06]
    for pt in pts:
        imagenet.presentation_time = pt
        imagenet.print_stats()

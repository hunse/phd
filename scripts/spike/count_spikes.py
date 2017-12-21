"""
Count number of spikes needed per image in my method and SotA competitors
"""
import numpy as np

from count_neuromorphic import Efficiency, mnist, cifar10, imagenet
import count_zambrano

# --- Zambrano
zneurons = count_zambrano.mnist.neurons_per_layer()
mnist_zamb = Efficiency(
    name='MNIST Zambrano',
    error=100 - 99.56,
    neurons=zneurons,
    synapses=[0] * (len(zneurons) + 1),
    rates=[67] * len(zneurons),
    presentation_time=0.291)
mnist_zamb_low = Efficiency(
    name='MNIST Zambrano Low FR',
    error=100 - 99.12,
    neurons=zneurons,
    synapses=[0] * (len(zneurons) + 1),
    rates=[10] * len(zneurons),
    presentation_time=0.209)

zneurons = count_zambrano.cifar10.neurons_per_layer()
cifar10_zamb = Efficiency(
    name='CIFAR-10 Zambrano',
    error=100 - 89.86,
    neurons=zneurons,
    synapses=[0] * (len(zneurons) + 1),
    rates=[68] * len(zneurons),
    presentation_time=0.372)
cifar10_zamb_low = Efficiency(
    name='CIFAR-10 Zambrano Low FR',
    error=100 - 88.52,
    neurons=zneurons,
    synapses=[0] * (len(zneurons) + 1),
    rates=[22] * len(zneurons),
    presentation_time=0.304)

zneurons = count_zambrano.imagenet.neurons_per_layer()
imagenet_zamb = Efficiency(
    name='Imagenet Zambrano',
    error=100 - 62.97,
    neurons=zneurons,
    synapses=[0] * (len(zneurons) + 1),
    rates=[66] * len(zneurons),
    presentation_time=0.347)
imagenet_zamb_low = Efficiency(
    name='Imagenet Zambrano Low FR',
    error=100 - 53.77,
    neurons=zneurons,
    synapses=[0] * (len(zneurons) + 1),
    rates=[12] * len(zneurons),
    presentation_time=0.338)

# print(cifar10.spikes_per_image())
# print(cifar10_zamb.spikes_per_image())

# print(imagenet.spikes_per_image())
# print(imagenet_zamb.spikes_per_image())
# print(imagenet_zamb_low.spikes_per_image())


def print_table_row(e, s='M'):
    spikes = e.spikes_per_image()
    values = (("%0.2f" if e.error < 10 else "%0.1f") % e.error,
              "%0.1f k" % (sum(e.neurons) / 1000.),
              "%0.1f" % e.average_rate(),
              int(e.presentation_time*1000),
              "%0.1f M" % (spikes / 1e6) if s == 'M' else
              "%0.0f k" % (spikes / 1e3))
    print(" & ".join(str(v) for v in values) + " \\\\")

print_table_row(mnist, s='k')
print_table_row(mnist_zamb, s='k')
print_table_row(mnist_zamb_low, s='k')

print_table_row(cifar10)
print_table_row(cifar10_zamb)
print_table_row(cifar10_zamb_low)

print_table_row(imagenet)
print_table_row(imagenet_zamb)
print_table_row(imagenet_zamb_low)

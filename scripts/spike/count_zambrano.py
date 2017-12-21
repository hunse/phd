"""
Count number of neurons in Zambrano (2017) networks.
"""
import numpy as np


class Network(object):
    def __init__(self, name):
        self.name = name
        self.shapes = []

    def add_input(self, m, n, c):
        assert len(self.shapes) == 0
        self.shapes.append(('input', m, n, c))

    def add_conv(self, n_filters, k_size):
        _, m, n, c = self.shapes[-1]
        self.shapes.append(('conv', m, n, n_filters))

    def add_pool(self, k_size):
        _, m, n, c = self.shapes[-1]
        m2 = int(np.ceil(float(m) / k_size))
        n2 = int(np.ceil(float(n) / k_size))
        self.shapes.append(('pool', m2, n2, c))

    def add_full(self, n_neurons):
        self.shapes.append(('full', 1, 1, n_neurons))

    def add_resblock(self, n_filters, k_size=3, stride=1):
        if stride > 1:
            self.add_pool(stride)
        self.add_conv(n_filters, k_size)
        self.add_conv(n_filters, k_size)

    def add_reslayer(self, n_filters, n_blocks, stride=1):
        self.add_resblock(n_filters, stride=stride)
        for _ in range(1, n_blocks):
            self.add_resblock(n_filters)

    def neurons_per_layer(self):
        layers = []
        for kind, m, n, c in self.shapes:
            layers.append(m*n*c if kind in ('conv', 'full') else 0)
        return layers

    def count_neurons(self):
        return sum(self.neurons_per_layer())

    def __str__(self):
        return "%s: n_neurons = %d" % (self.name, self.count_neurons())


mnist = Network('MNIST')
mnist.add_input(28, 28, 1)
mnist.add_conv(64, 3)
mnist.add_pool(2)
mnist.add_conv(128, 3)
mnist.add_conv(128, 3)
mnist.add_pool(2)
mnist.add_full(256)
mnist.add_full(50)

cifar10 = Network('CIFAR-10')
cifar10.add_input(32, 32, 3)
cifar10.add_conv(64, 3)
cifar10.add_conv(64, 3)
cifar10.add_pool(2)
cifar10.add_conv(128, 3)
cifar10.add_conv(128, 3)
cifar10.add_pool(2)
cifar10.add_conv(256, 3)
cifar10.add_conv(256, 3)
cifar10.add_conv(256, 3)
cifar10.add_pool(2)
cifar10.add_conv(512, 3)
cifar10.add_conv(512, 3)
cifar10.add_conv(512, 3)
cifar10.add_pool(2)
cifar10.add_full(512)

cifar100 = Network('CIFAR-100')
cifar100.add_input(32, 32, 3)
cifar100.add_conv(64, 3)
cifar100.add_conv(64, 3)
cifar100.add_pool(2)
cifar100.add_conv(128, 3)
cifar100.add_conv(128, 3)
cifar100.add_pool(2)
cifar100.add_conv(256, 3)
cifar100.add_conv(256, 3)
cifar100.add_conv(256, 3)
cifar100.add_pool(2)
cifar100.add_conv(1024, 3)
cifar100.add_conv(1024, 3)
cifar100.add_conv(1024, 3)
cifar100.add_pool(2)
cifar100.add_full(1024)

imagenet = Network('ImageNet')
imagenet.add_input(224, 224, 3)
imagenet.add_reslayer(64, 2, stride=1)
imagenet.add_reslayer(128, 2, stride=2)
imagenet.add_reslayer(256, 2, stride=2)
imagenet.add_reslayer(512, 2, stride=2)
imagenet.add_pool(7)
imagenet.add_full(512)


if __name__ == '__main__':
    print(mnist)
    print(cifar10)
    print(cifar100)
    print(imagenet)

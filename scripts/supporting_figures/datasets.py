import matplotlib.pyplot as plt
import numpy as np

import nengo_extras.data
from nengo_extras.matplotlib import tile

plt.figure(figsize=(6.4, 6.4), dpi=200)

plt.subplot(511)
(mnist, _), _ = nengo_extras.data.load_mnist()
tile(mnist.reshape(-1, 28, 28), rows=3, cols=20)
plt.title('MNIST')

plt.subplot(512)
(svhn, _), _ = nengo_extras.data.load_svhn()
tile(np.transpose(svhn.reshape(-1, 3, 32, 32), (0, 2, 3, 1)), rows=3, cols=20)
plt.title('SVHN')

plt.subplot(513)
(cifar10, _), _ = nengo_extras.data.load_cifar10(n_train=1, n_test=0)
tile(np.transpose(cifar10.reshape(-1, 3, 32, 32), (0, 2, 3, 1)), rows=3, cols=20)
plt.title('CIFAR-10')

plt.subplot(514)
(cifar100, _), _ = nengo_extras.data.load_cifar100()
tile(np.transpose(cifar100.reshape(-1, 3, 32, 32), (0, 2, 3, 1)), rows=3, cols=20)
plt.title('CIFAR-100')

plt.subplot(515)
ilsvrc2012, _, _, _ = nengo_extras.data.load_ilsvrc2012(n_files=1)
tile(np.transpose(ilsvrc2012.reshape(-1, 3, 256, 256), (0, 2, 3, 1)), rows=2, cols=14)
plt.title('ILSVRC-2012')


plt.tight_layout()

# plt.savefig('datasets.pdf')
plt.savefig('datasets.png')

plt.show()

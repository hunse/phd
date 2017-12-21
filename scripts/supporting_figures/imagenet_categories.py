"""Show example images from specified imagenet categories
"""

import matplotlib.pyplot as plt
import numpy as np

import nengo_extras.data
from nengo_extras.matplotlib import tile

images, labels, data_mean, label_names = nengo_extras.data.load_ilsvrc2012()

# images, labels, data_mean, label_names = (
#     nengo_extras.data.load_ilsvrc2012(n_files=1))

for s in label_names:
    print(s)

plt.figure()
# show = 'butcher'
show_names = ['butcher', 'restaurant', 'planetarium', 'church', 'library']

for i, show_name in enumerate(show_names):
    match = [s.startswith(show_name) for s in label_names]
    index = match.index(True) if True in match else None
    mask = (labels == index) if index is not None else None
    count = mask.sum() if mask is not None else None
    if count is not None:
        print("Found %d matches for %r: showing index %d, %d examples" % (
            sum(match), show_name, index, count))

        plt.subplot(len(show_names), 1, i+1)
        tile(np.transpose(images[mask].reshape(-1, 3, 256, 256), (0, 2, 3, 1)),
             rows=3, cols=5)

plt.show()

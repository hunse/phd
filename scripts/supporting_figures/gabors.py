import matplotlib.pyplot as plt
import numpy as np

from nengo.dists import Uniform
from nengo_extras.matplotlib import tile
from nengo_extras.vision import Gabor
from hunse_thesis.dists import LogUniform

rng = np.random.RandomState(3)

# r, c = 10, 20
r, c = 9, 12

# gabor = Gabor()
# gabor = Gabor(freq=Uniform(0.5, 1.5))
gabor = Gabor(freq=LogUniform(np.log(0.5), np.log(1.5)))

gabors = gabor.generate(r*c, (32, 32), rng=rng)

tile(gabors, rows=r, cols=c)

plt.savefig('gabors.pdf')
plt.show()

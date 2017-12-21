"""
Approximate firing rate of neurons in Cao et al. 2014 network
"""

import numpy as np

sizes = [80, 14, 1, 1]
filters = [8, 32, 128, 6]

neurons = sum(s**2 * f for s, f in zip(sizes, filters))
spikes = 5e5
time = 0.1

print(neurons)
print(float(spikes) / neurons / time)

# sizes

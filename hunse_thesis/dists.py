import numpy as np

from nengo.dists import Distribution
from nengo.params import NumberParam


class LogGaussian(Distribution):

    log_mean = NumberParam('log_mean')
    log_std = NumberParam('log_std')
    base = NumberParam('base', low=0, low_open=True)

    def __init__(self, log_mean, log_std, base=np.e):
        super(LogGaussian, self).__init__()
        self.log_mean = log_mean
        self.log_std = log_std
        self.base = base

    def __repr__(self):
        return "LogGaussian(log_mean=%r, log_std=%r)" % (
            self.log_mean, self.log_std)

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        log = rng.normal(loc=self.log_mean, scale=self.log_std, size=shape)
        return self.base**log


class LogUniform(Distribution):

    log_low = NumberParam('log_low')
    log_high = NumberParam('log_high')
    base = NumberParam('base', low=0, low_open=True)

    def __init__(self, log_low, log_high, base=np.e):
        super(LogUniform, self).__init__()
        self.log_low = log_low
        self.log_high = log_high
        self.base = base

    def __repr__(self):
        return "LogUniform(log_low=%r, log_high=%r)" % (
            self.log_low, self.log_high)

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        log = rng.uniform(self.log_low, self.log_high, size=shape)
        return self.base**log

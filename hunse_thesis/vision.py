import numpy as np


def percentile_encoders_intercepts(
        trainX, encoders, percentile=50, norm='std'):
    trainX = trainX.reshape(trainX.shape[0], -1)
    H = np.dot(trainX, encoders.T)
    intercepts = np.percentile(H, percentile, axis=0)
    return scale_encoders_intercepts(encoders, intercepts, norm=norm, H=H)


def scale_encoders_intercepts(
        encoders, intercepts, norm=None, scale=1., trainX=None, H=None):
    assert H is not None or trainX is not None
    if H is None:
        trainX = trainX.reshape(trainX.shape[0], -1)
        H = np.dot(trainX, encoders.T)

    if norm == 'std':
        r = H.std(axis=0)
        encoders = encoders / r[:, None]
        intercepts = intercepts / r
    elif norm == 'range':
        r = 0.25 * (H.max(axis=0) - H.min(axis=0))
        encoders = encoders * (scale / r[:, None])
        intercepts = intercepts * (scale / r)
    elif norm is not None:
        raise ValueError("Unrecognized norm type %r" % norm)

    return encoders, intercepts

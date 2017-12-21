import numpy as np

import nengo
from nengo.utils.compat import is_iterable
from nengo.utils.numpy import rms, norm


class EAIO(nengo.Network):
    """Give EnsembleArray API to Ensemble or Node.

    This allows Node/Ensemble/EnsembleArray to be used more interchangably.
    """

    def __init__(self, cls, *args, **kwargs):
        label = kwargs.pop('label', None)
        seed = kwargs.pop('seed', None)
        add_to_container = kwargs.pop('add_to_container', None)
        super(EAIO, self).__init__(label, seed, add_to_container)

        with self:
            self._obj = cls(*args, **kwargs)

    @property
    def input(self):
        return self._obj

    @property
    def output(self):
        return self._obj

    @property
    def neuron_input(self):
        return getattr(self._obj, 'neurons', None)

    @property
    def neuron_output(self):
        return getattr(self._obj, 'neurons', None)


def angle(x, y, axis=-1):
    """Angle between vectors along axis"""
    eps = np.finfo(x.dtype).tiny
    return np.arccos((x * y).sum(axis=axis) / (
        np.sqrt((x**2).sum(axis=axis) * (y**2).sum(axis=axis)) + eps))


def initial_w(shape, kind='', normkind=None, offset=0.0, scale=1.0, rng=np.random):
    assert len(shape) == 2
    kind = kind.lower()
    if kind == 'zeros':
        w = np.zeros(shape)
    elif kind == 'identity':
        min_shape = min(shape)
        ceili = lambda s: int(np.ceil(float(s) / min_shape))
        w = np.tile(np.eye(min_shape), (ceili(shape[0]), ceili(shape[1])))[
            :shape[0], :shape[1]]
        w -= w.mean()
    elif kind == 'binary':
        w = 2.*rng.randint(0, 2, size=shape) - 1.
    elif kind == 'uniform':
        w = rng.uniform(-1, 1, size=shape)
    elif kind == 'norm_uniform':
        w = rng.uniform(-1, 1, size=shape)
        w /= norm(w)
    elif kind in ['normal', 'gaussian']:
        w = rng.normal(size=shape)
    elif kind in ['norm_normal', 'norm_gaussian']:
        w = rng.normal(size=shape)
        w /= norm(w)
    elif kind == 'ortho':
        # w = orthogonalize(rng.uniform(-1, 1, size=shape))
        w = orthogonalize(rng.normal(size=shape))
    elif kind == 'orthonorm':
        # w = orthogonalize(rng.uniform(-1, 1, size=shape))
        w = orthogonalize(rng.normal(size=shape))
        w /= norm(w)
    else:
        raise ValueError("Unrecognized kind %r" % kind)

    # normalixe (let X ~ N(0, 1))
    if normkind == 'left':  # then WX ~ N(0, 1)
        w /= norm(w, axis=1, keepdims=True)
    elif normkind == 'right':  # then XW ~ N(0, 1)
        w /= norm(w, axis=0, keepdims=True)
    if normkind == 'leftmean':  # then WX ~ N(0, 1)
        w /= norm(w, axis=1).mean()
    elif normkind == 'rightmean':  # then XW ~ N(0, 1)
        w /= norm(w, axis=0).mean()
    elif normkind:
        raise ValueError("Unrecognized norm kind %r" % normkind)

    if scale != 1.0:
        w *= scale
    if offset != 0.0:
        w += offset
    return w


def initial_weights(sizes, **kwargs):
    return [initial_w((s0, s1), **kwargs) for s0, s1 in zip(sizes, sizes[1:])]


def orthogonalize(X):
    """Orthogonalize matrix by making singular values 1"""
    assert X.ndim == 2
    U, s, V = np.linalg.svd(X, full_matrices=False)
    return np.dot(U, V)


def neural_activities(sim, ens, x):
    """Calculates the neural activities of an ensemble.
    """
    if x.ndim == 1:
        x = x.reshape((1, -1))
    assert x.ndim == 2 and x.shape[1] == ens.n_neurons

    built_ens = sim.data[ens]
    y = ens.neuron_type.rates(x, built_ens.gain, built_ens.bias)
    return y


def nengoobj_io(nengo_object, attr):
    if isinstance(nengo_object, (nengo.Ensemble, nengo.Node)):
        return nengo_object
    elif isinstance(nengo_object, nengo.Network):
        if hasattr(nengo_object, attr):
            return getattr(nengo_object, attr)
        else:
            raise ValueError("Nengo network has no attribute %r" % attr)


def nengoobj_in(nengo_object, attr='input'):
    return nengoobj_io(nengo_object, attr=attr)


def nengoobj_out(nengo_object, attr='output'):
    return nengoobj_io(nengo_object, attr=attr)


def lsuv(X, ws, fs, **kwargs):
    """Layer-sequential unit-variance initialization [1]_

    References
    ----------
    .. [1] Mishkin, D., & Matas, J. (2016). All you need is a good init.
       In ICLR 2016 (pp. 1-13).
    """
    fs = ([fs] * (len(ws) - 1) + [None]) if not is_iterable(fs) else fs
    assert len(ws) >= 2
    assert len(fs) == len(ws)

    for i, (w, f) in enumerate(zip(ws, fs)):
        X = lsuv_layer(X, w, f, layer_i=i, **kwargs)


def lsuv_layer(X, w, f, t_max=50, target_std=1., target_input=False, atol=1e-2,
               layer_i=None, verbose=0):
    name = "Layer%s" % (" %d" % layer_i if layer_i is not None else "")

    Y = np.dot(X, w)

    if f is None or target_input:
        Ystd = Y.std(axis=1).mean()
        # assert Ystd > atol
        assert Ystd > 1e-8
        w *= (target_std / Ystd)
        Y *= (target_std / Ystd)
        return Y if f is None else f(Y)

    for i in range(t_max):
        Z = f(Y)
        Zstd = Z.std(axis=1).mean()
        assert Zstd > atol
        if verbose >= 1:
            print("  Iteration %d: std=%0.3f" % (i, Zstd))
        if abs(Zstd - target_std) < atol:
            break

        w *= (target_std / Zstd)
        Y *= (target_std / Zstd)  # just scale, no need to do dot again
    else:
        if verbose >= 0:
            print("%s did not converge after %d iterations (std=%0.3e)"
                  % (name, t_max, Zstd))

    return Z

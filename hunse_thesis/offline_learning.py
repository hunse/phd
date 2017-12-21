import numpy as np

from nengo.utils.compat import is_array_like, is_number, is_iterable
from nengo.utils.numpy import rms

from .utils import angle, norm, orthogonalize


def make_flat_batch_fn(X, Y, batch_size):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] % batch_size == 0
    n_batches = X.shape[0] // batch_size
    XX = X.reshape((n_batches, batch_size, -1))
    YY = Y.reshape((n_batches, batch_size, -1))

    def fn(epoch):
        return zip(XX, YY)

    return fn


def make_random_batch_fn(X, Y, batch_size, rng=np.random):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] % batch_size == 0
    n = X.shape[0]
    n_batches = n // batch_size
    X = X.reshape((n, -1))
    Y = Y.reshape((n, -1))

    def fn(epoch):
        inds = rng.permutation(n).reshape(n_batches, -1)
        for i in inds:
            yield X[i], Y[i]

    return fn


def squared_cost(y, ystar):
    """ squared error, C = 0.5 * (y - ystar)**2
    """
    assert y.ndim == ystar.ndim == 2
    C = 0.5 * ((y - ystar)**2).sum(axis=1)
    dC = y - ystar
    return C, dC


def squared_error(y, ystar):
    return squared_cost(y, ystar)[0]


def rms_error(y, ystar):
    assert y.ndim == ystar.ndim == 2
    return rms(y - ystar, axis=1)


def squared_cost_on_inds(y, yinds):
    assert y.ndim == 2
    assert yinds.ndim == 1 or yinds.ndim == 2 and yinds.shape[1] == 1
    dC = np.array(y)
    dC[np.arange(len(y)), yinds.ravel()] -= 1
    C = 0.5 * (dC**2).sum(axis=1)
    return C, dC


def rms_error_on_inds(y, yinds):
    assert y.ndim == 2
    assert yinds.ndim == 1 or yinds.ndim == 2 and yinds.shape[1] == 1
    yd = np.array(y)
    yd[np.arange(len(y)), yinds.ravel()] -= 1
    return rms(yd, axis=1)


def nll_cost_on_inds(y, yinds):
    """ nll error, C = -log softmax(y)[yind]

        dC/dy_i = softmax(y) - (1 if i == yind else 0)
    """
    assert y.ndim == 2
    assert yinds.ndim == 1 or yinds.ndim == 2 and yinds.shape[1] == 1
    ey = np.exp(y - y.max(axis=1, keepdims=1))
    z = ey / ey.sum(axis=1, keepdims=1)
    # C = -np.log(z[np.arange(len(y)), yinds.ravel()])
    C = -np.log(np.maximum(z[np.arange(len(y)), yinds.ravel()], 1e-16))
    dC = np.array(z)
    dC[np.arange(len(y)), yinds.ravel()] -= 1
    return C, dC


def class_error_on_inds(y, yinds):
    assert y.ndim == 2
    assert yinds.ndim == 1 or yinds.ndim == 2 and yinds.shape[1] == 1
    return (np.argmax(y, axis=-1) != yinds.ravel())


def pointer_squared_cost_on_inds(y, yinds, pointers):
    assert yinds.ndim == 1 or yinds.ndim == 2 and yinds.shape[1] == 1
    ystar = pointers[yinds.ravel()]
    return squared_cost(y, ystar)


def pointer_class_error_on_inds(y, yinds, pointers):
    return class_error_on_inds(np.dot(y, pointers.T), yinds)


def layer_scalar(param, n_layers):
    if is_number(param):
        param = [param] * n_layers
    assert (is_iterable(param) and
            all(is_number(v) for v in param) and
            len(param) == n_layers)
    return param


def batch_layer_scalar(param, t, n_layers):
    if callable(param):
        param = param(t)
    return layer_scalar(param, n_layers)


def momentum_update(ps, dps, vps=None, momentums=0):
    if vps is None:
        vps = [None] * len(ps)

    for p, dp, vp, momentum in zip(ps, dps, vps, momentums):
        if None not in (p, dp):
            if vps is not None and momentum != 0:
                vp *= momentum
                vp += dp
                p += vp
            else:
                p += dp


class Network(object):
    def __init__(self, weights, biases=None, backweights=None, backbiases=None,
                 f=None, df=None, g=None, dg=None, noise=0,
                 weight_masks=None):
        self.weights = [np.array(W, dtype=float) for W in weights]
        self.biases = self._convert_biases(biases, self.weights)
        self.f = f
        self.df = df
        self.noise = noise

        # --- backwards parameters
        self.backweights = (
            [None] * self.n_layers if backweights is None else
            [np.array(W, dtype=float) for W in backweights])
        self.backbiases = self._convert_biases(backbiases, self.backweights)
        self.g = f if g is None else g
        self.dg = df if dg is None else dg

        # --- masks
        self.weight_masks = [] if weight_masks is None else weight_masks
        assert all(m.shape == w.shape
                   for w, m in zip(self.weights, self.weight_masks))
        self.mask_weights()

    def _convert_biases(self, biases, weights):
        if biases is None:
            return [None] * len(weights)
        elif is_number(biases):
            return [biases * np.ones(W.shape[1], dtype=float)
                    for W in weights]
        elif is_iterable(biases) and all(is_array_like(b) for b in biases):
            biases = [np.array(b, dtype=float) for b in biases]
            assert len(biases) == len(weights)
            assert all(b.ndim == 1 and b.size == w.shape[1]
                       for b, w in zip(biases, weights))
            return biases
        else:
            raise ValueError("Cannot convert 'biases'")

    @property
    def n_layers(self):
        return len(self.weights)

    @property
    def sizes(self):
        return [w.shape[0] for w in self.weights] + [self.weights[-1].shape[1]]

    def forward(self, x, rng=np.random):
        acts, outs = [x], [x]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            acts.append(np.dot(outs[-1], W))
            if b is not None:
                acts[-1] += b
            if self.noise:
                acts[-1] += rng.normal(scale=self.noise, size=acts[-1].shape)
            outs.append(self.f(acts[-1]))

        # final layer has no nonlinearity
        acts.append(np.dot(outs[-1], self.weights[-1]))
        if self.biases[-1] is not None:
            acts[-1] += self.biases[-1]
        outs.append(np.array(acts[-1]))

        return acts, outs

    def predict(self, x):
        acts, outs = self.forward(x)
        return outs[-1]

    def forward1(self, i, x, rng=np.random):
        W, b = self.weights[i], self.biases[i]
        a = np.dot(x, W)
        if b is not None:
            a += b
        if self.noise:
            a += rng.normal(scale=self.noise, size=a.shape)
        return a, (self.g(a) if i < self.n_layers - 1 else a)

    def back1(self, i, x):
        V, c = self.backweights[i], self.backbiases[i]
        a = np.dot(x, V)
        if c is not None:
            a += c
        return a, (self.g(a) if i > 0 else a)

    def mask_weights(self):
        for W, Wm in zip(self.weights, self.weight_masks):
            W *= Wm


class Learner(object):
    def __init__(self, network, cost, error, eta=1e-2, alpha=0, momentum=0,
                 rng=np.random, name=None):
        self.network = network
        self.cost = cost
        self.error = error
        self.eta = eta  # learning rate
        self.alpha = alpha  # weight decay
        self.rng = rng
        self.name = name

        self.momentum = momentum
        self.vWs = None
        self.vbs = None
        self.bvWs = None
        self.bvbs = None

        self.batch_costs = []
        self.batch_errors = []
        self.train_costs = []
        self.train_errors = []
        self.test_errors = []

        self.weight_norms = None
        self.delta_norms = None
        self.dW_norms = None
        self.db_norms = None

        self.check_finite = False

    def __str__(self):
        return type(self).__name__ if self.name is None else str(self.name)

    def train(self, epochs, batch_fn, test_set=None, verbose=1):
        n_layers = self.network.n_layers

        Ws, bs = self.network.weights, self.network.biases
        for epoch in range(epochs):
            batches = list(batch_fn(epoch))
            n_batches = len(batches)

            batch_costs = []
            batch_errors = []
            for i_batch, [x, ystar] in enumerate(batches):
                t = epoch + float(i_batch) / n_batches
                acts, outs = self.network.forward(x)

                C, dC = self.cost(outs[-1], ystar)
                batch_costs.append(C.mean())

                E = self.error(outs[-1], ystar)
                batch_errors.append(E.mean())

                dWs, dbs = self.get_deltas(acts, outs, dC)
                if self.dW_norms is not None:
                    self.dW_norms.append([norm(d) for d in dWs])
                if self.db_norms is not None:
                    self.db_norms.append([norm(d) for d in dbs])

                self.update_weights_biases(t, dWs, dbs)

                # bdWs, bdbs = self.get_back_deltas(acts, outs, dC)
                # self.update_backweights_backbiases(t, bdWs, bdbs)

                self.update(t, acts, outs, dC)

                if self.check_finite:
                    for W in self.network.weights:
                        assert np.isfinite(W).all()
                    for W in self.network.backweights:
                        assert np.isfinite(W).all()

                if self.weight_norms is not None:
                    self.weight_norms.append([norm(W) for W in self.network.weights])

                if verbose >= 2:
                    print("Cost/error %d-%d: %0.2e, %0.2e" % (
                        epoch, i_batch, batch_costs[-1], batch_errors[-1]))

            self.batch_costs.extend(batch_costs)
            self.batch_errors.extend(batch_errors)
            self.train_costs.append(np.mean(batch_costs))
            self.train_errors.append(np.mean(batch_errors))

            if test_set:
                x, ystar = test_set
                _, outs = self.network.forward(x)
                errors = self.error(outs[-1], ystar)
                self.test_errors.append(errors.mean())

                silents = [(out.max(axis=0) == 0).mean() for out in outs[1:-1]]
                if verbose >= 1:
                    print("Epoch %d: cost=%0.2e, train=%0.3f, test=%0.3f"
                          " (silent=%s)" % (
                              epoch, self.train_costs[-1],
                              self.train_errors[-1], self.test_errors[-1],
                              ", ".join("%0.1f%%" % (100*s) for s in silents)
                          ))
            elif verbose >= 1:
                print("Epoch %d: cost=%0.2e, train=%0.3f" % (
                    epoch, self.train_costs[-1], self.train_errors[-1]))

    def test(self, dataset):
        x, ystar = dataset
        _, outs = self.network.forward(x)
        errors = self.error(outs[-1], ystar)
        return errors

    def get_deltas(self, acts, outs, dC):
        raise NotImplementedError()

    def get_back_deltas(self, acts, outs, dC):
        n_layers = self.network.n_layers
        return [None] * n_layers, [None] * n_layers

    def update(self, t, acts, outs, dC):
        pass  # implement for custom updating, e.g. adaptive FA

    def _update_param(self, Ws, vWs, dWs, etas, momentums=None, alphas=None):
        if momentums is None:
            momentums = [0] * len(etas)
        if alphas is None:
            alphas = [0] * len(etas)

        for W, vW, dW, eta, momentum, alpha in zip(
                Ws, vWs, dWs, etas, momentums, alphas):
            if W is not None and dW is not None:
                if momentum != 0:
                    vW *= momentum
                    vW += dW
                else:
                    vW = dW
                if alpha != 0:
                    vW = vW - alpha * W
                W += eta * vW

    def _update_weights_biases(
            self, Ws, bs, vWs, vbs, dWs, dbs, etas, momentums, alphas):
        self._update_param(Ws, vWs, dWs, etas, momentums, alphas)
        self._update_param(bs, vbs, dbs, etas, momentums)

    def update_weights_biases(self, t, dWs, dbs):
        etas = batch_layer_scalar(self.eta, t, self.network.n_layers)
        momentums = batch_layer_scalar(self.momentum, t, self.network.n_layers)
        alphas = batch_layer_scalar(self.alpha, t, self.network.n_layers)

        Ws, bs = self.network.weights, self.network.biases

        if self.vWs is None:
            self.vWs = [np.zeros_like(W) for W in Ws]
        if self.vbs is None:
            self.vbs = [None if b is None else np.zeros_like(b) for b in bs]

        self._update_weights_biases(
            Ws, bs, self.vWs, self.vbs, dWs, dbs, etas, momentums, alphas)

        self.network.mask_weights()

    def update_backweights_backbiases(self, t, dWs, dbs):
        etas = batch_layer_scalar(self.beta, t, self.network.n_layers)
        momentums = batch_layer_scalar(self.bmomentum, t, self.network.n_layers)
        alphas = batch_layer_scalar(self.balpha, t, self.network.n_layers)

        Ws, bs = self.network.backweights, self.network.backbiases

        if self.bvWs is None:
            self.bvWs = [np.zeros_like(W) for W in Ws]
        if self.bvbs is None:
            self.bvbs = [None if b is None else np.zeros_like(b) for b in bs]

        self._update_weights_biases(
            Ws, bs, self.bvWs, self.bvbs, dWs, dbs, etas, momentums, alphas)

        self.network.mask_weights()


class ShallowLearner(Learner):
    def get_deltas(self, acts, outs, dC):
        Ws, bs = self.network.weights, self.network.biases
        dWs = [None] * len(Ws)
        dbs = [None] * len(bs)
        dWs[-1] = -np.dot(outs[-2].T, dC)
        dbs[-1] = -dC.sum(0)
        return dWs, dbs


class BPLearner(Learner):
    def get_deltas(self, acts, outs, dC):
        deltas = [dC]
        Ws, bs = self.network.weights, self.network.biases

        dWs = []
        dbs = []
        delta_norms = [] if self.delta_norms is not None else None
        for a, v, W, b in list(zip(acts, outs, Ws, bs))[::-1]:
            delta = deltas[-1]
            dWs.append(-np.dot(v.T, delta))
            dbs.append(-delta.sum(0))
            deltas.append(np.dot(delta, W.T) * self.network.df(a))
            if delta_norms is not None:
                delta_norms.append(norm(delta, axis=1).mean())

        if self.delta_norms is not None:
            self.delta_norms.append(delta_norms[::-1])

        return dWs[::-1], dbs[::-1]


class BPLocalLearner(Learner):
    def get_deltas(self, acts, outs, dC):
        Ws, bs = self.network.weights, self.network.biases

        dWs = []
        dbs = []
        delta_norms = [] if self.delta_norms is not None else None
        delta = dC
        da = np.ones_like(outs[-1])
        for a, v, W, b in list(zip(acts, outs, Ws, bs))[::-1]:
            delta_da = delta * da
            dWs.append(-np.dot(v.T, delta_da))
            dbs.append(-delta_da.sum(0))
            if delta_norms is not None:
                delta_norms.append(norm(delta_da, axis=1).mean())

            delta = np.dot(delta, W.T)
            da = self.network.df(a)

        if self.delta_norms is not None:
            self.delta_norms.append(delta_norms[::-1])

        return dWs[::-1], dbs[::-1]


class BPLocalLearner2(Learner):
    def get_deltas(self, acts, outs, dC):
        Ws, bs = self.network.weights, self.network.biases

        dWs = []
        dbs = []
        delta_norms = [] if self.delta_norms is not None else None
        delta = dC
        da = np.ones_like(outs[-1])
        for a, v, W, b in list(zip(acts, outs, Ws, bs))[::-1]:
            delta_da = delta * da
            dWs.append(-np.dot(v.T, delta_da))
            dbs.append(-delta_da.sum(0))
            if delta_norms is not None:
                delta_norms.append(norm(delta_da, axis=1).mean())

            da = self.network.df(a)
            dascale = da[da > 0].mean()
            # dascale = np.percentile(da[da > 0], 50)
            delta = np.dot(delta * dascale, W.T)

        if self.delta_norms is not None:
            self.delta_norms.append(delta_norms[::-1])

        return dWs[::-1], dbs[::-1]


class FALearner(Learner):
    """Feedback Alignment

    Uses random B matrices to propagate error backwards, where each B matrix
    corresponds to one set of weights.
    """

    def __init__(self, network, cost, error, betas=0.5, **kwargs):
        Learner.__init__(self, network, cost, error, **kwargs)

        if betas is None:
            self.Bs = []  # user must define these later
        else:
            betas = layer_scalar(betas, self.network.n_layers - 1)
            size_out = self.network.weights[-1].shape[1]
            self.Bs = [self.rng.uniform(-beta, beta, size=(W.shape[1], W.shape[0]))
                       for W, beta in zip(self.network.weights[1:], betas)]

        self.bp_angles = None  # set to [] to record
        self.pbp_angles = None  # set to [] to record
        self.bpd_angles = None  # set to [] to record, angle with full bp delta

    def bp_angle(self, deltah, e, a, W):
        deltah_bp = np.dot(e, W.T) * self.network.df(a)
        return angle(deltah_bp, deltah).mean(0)

    def pbp_angle(self, deltah, e, a, W):
        deltah_pbp = np.dot(e, np.linalg.pinv(W)) * self.network.df(a)
        return angle(deltah_pbp, deltah).mean(0)

    def get_deltas(self, acts, outs, dC):
        Ws, bs = self.network.weights, self.network.biases

        deltas = [dC]
        Bs = [None] + self.Bs

        dWs = []
        dbs = []
        delta_norms = [] if self.delta_norms is not None else None
        bp_angles = [] if self.bp_angles is not None else None
        pbp_angles = [] if self.pbp_angles is not None else None
        bpd_angles = [] if self.bpd_angles is not None else None
        bp_delta = deltas[-1]
        for a, v, W, b, B in list(zip(acts, outs, Ws, bs, Bs))[::-1]:
            delta = deltas[-1]
            dWs.append(-np.dot(v.T, delta))
            dbs.append(-delta.sum(0))
            da = self.network.df(a)
            deltas.append(np.dot(delta, B) * da if B is not None else None)
            if delta_norms is not None:
                delta_norms.append(norm(delta, axis=1).mean())

            # compute angles
            if deltas[-1] is not None and bp_angles is not None:
                bp_angles.append(self.bp_angle(deltas[-1], delta, a, W))
            if deltas[-1] is not None and pbp_angles is not None:
                pbp_angles.append(self.pbp_angle(deltas[-1], delta, a, W))
            if deltas[-1] is not None and bpd_angles is not None:
                bp_delta = np.dot(bp_delta, W.T) * da
                bpd_angles.append(angle(bp_delta, deltas[-1]).mean(0))

        if self.delta_norms is not None:
            self.delta_norms.append(delta_norms[::-1])
        if self.bp_angles is not None:
            self.bp_angles.append(bp_angles[::-1])
        if self.pbp_angles is not None:
            self.pbp_angles.append(pbp_angles[::-1])
        if self.bpd_angles is not None:
            self.bpd_angles.append(bpd_angles[::-1])

        return dWs[::-1], dbs[::-1]


class FALocalLearner(FALearner):
    def get_deltas(self, acts, outs, dC):
        Ws, bs = self.network.weights, self.network.biases

        deltas = [dC]
        Bs = [None] + self.Bs

        dWs = []
        dbs = []
        delta_norms = [] if self.delta_norms is not None else None
        bp_angles = [] if self.bp_angles is not None else None
        pbp_angles = [] if self.pbp_angles is not None else None
        bpd_angles = [] if self.bpd_angles is not None else None
        da = np.ones_like(outs[-1])
        bp_delta = deltas[-1]
        for a, v, W, b, B in list(zip(acts, outs, Ws, bs, Bs))[::-1]:
            delta = deltas[-1] * da
            dWs.append(-np.dot(v.T, delta))
            dbs.append(-delta.sum(0))
            deltas.append(np.dot(deltas[-1], B) if B is not None else None)
            if delta_norms is not None:
                delta_norms.append(norm(delta, axis=1).mean())

            da = self.network.df(a)

            # compute angles
            if deltas[-1] is not None and bp_angles is not None:
                bp_angles.append(self.bp_angle(deltas[-1] * da, deltas[-2], a, W))
            if deltas[-1] is not None and pbp_angles is not None:
                pbp_angles.append(self.pbp_angle(deltas[-1] * da, deltas[-2], a, W))
            if deltas[-1] is not None and bpd_angles is not None:
                bp_delta = np.dot(bp_delta, W.T) * da
                bpd_angles.append(angle(bp_delta, deltas[-1] * da).mean(0))

        if self.delta_norms is not None:
            self.delta_norms.append(delta_norms[::-1])
        if self.bp_angles is not None:
            self.bp_angles.append(bp_angles[::-1])
        if self.pbp_angles is not None:
            self.pbp_angles.append(pbp_angles[::-1])
        if self.bpd_angles is not None:
            self.bpd_angles.append(bpd_angles[::-1])

        return dWs[::-1], dbs[::-1]


class FASkipLearner(Learner):
    """Feedback Alignment with skipping
    """

    def __init__(self, network, cost, error, betas=0.5, **kwargs):
        Learner.__init__(self, network, cost, error, **kwargs)

        if betas is None:
            self.Bs = []  # user must define these later
        else:
            betas = layer_scalar(betas, self.network.n_layers - 1)
            size_out = self.network.weights[-1].shape[1]
            self.Bs = [self.rng.uniform(-beta, beta, size=(W.shape[1], W.shape[0]))
                       for W, beta in zip(self.network.weights[1:], betas)]

        self.bpd_angles = None  # set to [] to record
        self.bpu_angles = None  # set to [] to record

    def get_deltas(self, acts, outs, dC):
        Ws, bs = self.network.weights, self.network.biases
        Bs = self.Bs + [None]
        da = None

        deltas = []
        dWs = []
        dbs = []
        delta_norms = [] if self.delta_norms is not None else None
        bpd_angles = [] if self.bpd_angles is not None else None
        bpu_angles = [] if self.bpu_angles is not None else None
        bp_delta = dC
        da = np.ones_like(outs[-1])
        for a, v, W, b, B in list(zip(acts, outs, Ws, bs, Bs))[::-1]:
            dCB = np.dot(dC, B) if B is not None else dC
            delta = dCB * da
            deltas.append(delta)
            dWs.append(-np.dot(v.T, delta))
            dbs.append(-delta.sum(0))
            if delta_norms is not None:
                delta_norms.append(norm(delta, axis=1).mean())
            if B is not None and bpd_angles is not None:
                bpd_angles.append(angle(da * bp_delta, delta).mean(0))
            if B is not None and bpu_angles is not None:
                bpu_angles.append(angle(bp_delta, dCB).mean(0))
            if bpd_angles is not None or bpu_angles is not None:
                bp_delta = np.dot(da * bp_delta, W.T)

            da = self.network.df(a)

        self._deltas = deltas[::-1]
        if self.delta_norms is not None:
            self.delta_norms.append(delta_norms[::-1])
        if self.bpd_angles is not None:
            self.bpd_angles.append(bpd_angles[::-1])
        if self.bpu_angles is not None:
            self.bpu_angles.append(bpu_angles[::-1])

        return dWs[::-1], dbs[::-1]


# class FAPopLearner(FASkipLearner):
#     """Using NEF populations in error channel
#     """
#     def __init__(self, *args, **kwargs):
#         from nengo.dists import UniformHypersphere

#         encoders = kwargs.pop('encoders', None)
#         intercepts = kwargs.pop('encoders', None)
#         FASkipLearner.__init__(self, *args, **kwargs)

#         if encoders is None:
#             d = self.network.layer_sizes[-1]
#             encoders = [UniformHypersphere(surface=True).sample(
#                 n, d, rng=self.rng).T for n in self.network.layer_sizes[1:-1]]
#         if intercepts is None:
#             intercepts = [0.01 * np.ones(s)
#                           for s in self.network.layer_sizes[1:-1]]

#         self.encoders = encoders
#         self.biases = [-x for x in intercepts]

#     def get_deltas(self, acts, outs, dC):
#         Ws, bs = self.network.weights, self.network.biases
#         Bs = self.Bs + [None]
#         da = None

#         deltas = []
#         dWs = []
#         dbs = []
#         delta_norms = [] if self.delta_norms is not None else None
#         bpd_angles = [] if self.bpd_angles is not None else None
#         bp_delta = dC
#         for a, v, W, b, B in list(zip(acts, outs, Ws, bs, Bs))[::-1]:
#             delta = (np.dot(dC, B) * da if B is not None else dC)
#             deltas.append(delta)
#             dWs.append(-np.dot(v.T, delta))
#             dbs.append(-delta.sum(0))
#             if delta_norms is not None:
#                 delta_norms.append(norm(delta, axis=1).mean())

#             da = self.network.df(a)

#             # compute angles
#             if bpd_angles is not None:
#                 if B is not None:
#                     bpd_angles.append(angle(bp_delta, delta).mean(0))
#                 bp_delta = np.dot(bp_delta, W.T) * da

#         self._deltas = deltas[::-1]
#         if self.delta_norms is not None:
#             self.delta_norms.append(delta_norms[::-1])
#         if self.bpd_angles is not None:
#             self.bpd_angles.append(bpd_angles[::-1])

#         return dWs[::-1], dbs[::-1]

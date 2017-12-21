import time

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.params import BoolParam, IntParam, NumberParam


def cho_solve(A, y, overwrite=False):
    # Solve A x = y for x
    try:
        import scipy.linalg
        factor = scipy.linalg.cho_factor(A, overwrite_a=overwrite)
        x = scipy.linalg.cho_solve(factor, y)
    except ImportError:
        L = np.linalg.cholesky(A)
        L = np.linalg.inv(L.T)
        x = np.dot(L, np.dot(L.T, y))

    return x


class Softmax(nengo.solvers.Solver):
    """Softmax solver. TODO
    """

    reg = NumberParam('reg', low=0)
    n_epochs = IntParam('n_epochs', low=1)
    verbose = IntParam('verbose')

    def __init__(self, reg=0, n_epochs=1000, weights=False, verbose=0):
        super(Softmax, self).__init__(weights=weights)
        self.reg = reg
        self.verbose = verbose
        self.n_epochs = n_epochs

    def __call__(self, A, Y, rng=None, E=None):
        from nengo_extras.convnet import softmax
        import scipy.optimize

        tstart = time.time()

        assert A.shape[0] == Y.shape[0]
        m, n = A.shape
        _, d = Y.shape
        Xshape = (n, d)

        # regularization
        sigma = self.reg * A.max()
        lamb = m * sigma**2

        # --- initialization
        X0 = np.zeros(Xshape)
        # X0 = rng.normal(scale=1./m, size=Xshape)
        # X0, _ = nengo.solvers.LstsqL2(reg=self.reg)(A, Y, rng=rng, E=E)

        # --- solve with L-BFGS
        yi = Y.argmax(axis=1)
        mi = np.arange(m)

        def f_df(x):
            X = x.reshape(Xshape)
            Yest = softmax(np.dot(A, X), axis=1)
            cost = -np.log(np.maximum(Yest[mi, yi], 1e-16)).sum()
            E = Yest - Y
            grad = np.dot(A.T, E)
            if lamb > 0:
                cost += 0.5 * lamb * (X**2).sum()
                grad += lamb * X
            return cost, grad.ravel()

        x0 = X0.ravel()
        x, mincost, info = scipy.optimize.fmin_l_bfgs_b(
            f_df, x0, maxfun=self.n_epochs, iprint=self.verbose)

        t = time.time() - tstart

        X = x.reshape(Xshape)
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(softmax(np.dot(A, X), axis=1) - Y, axis=1),
            'time': t,
            'iterations': info['funcalls'],
        }


class SoftmaxSGD(Softmax):
    eta = NumberParam('eta', low=0, low_open=True)
    momentum = NumberParam('momentum', low=-1, high=1)
    batch_size = IntParam('batch_size', low=1)

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop('eta', 2e-4)
        self.momentum = kwargs.pop('momentum', -0.9)
        self.batch_size = kwargs.pop('batch_size', 100)
        kwargs.setdefault('n_epochs', 25)
        super(SoftmaxSGD, self).__init__(*args, **kwargs)

    def __call__(self, A, Y, rng=None, E=None, X=None):
        from nengo_extras.convnet import softmax

        assert E is None
        assert A.ndim == Y.ndim == 2 and A.shape[0] == Y.shape[0]
        m = A.shape[0]
        Xshape = (A.shape[1], Y.shape[1])

        batch_size = self.batch_size

        # regularization
        sigma = self.reg * A.max()
        lamb = batch_size * sigma**2
        print("sigma^2: %s" % sigma**2)

        tstart = time.time()
        Y = self.mul_encoders(Y, E)

        # --- solve with SGD
        Yi = Y.argmax(axis=1)
        eta = self.eta
        momentum = self.momentum

        bi = np.arange(batch_size)

        def batches():
            for i in range(m // batch_size):
                r = range(i*batch_size, (i+1)*batch_size)
                yield A[r], Yi[r]

        def f_df(a, X, yi):
            yest = softmax(np.dot(a, X), axis=1)
            cost = -np.log(np.maximum(yest[bi, yi], 1e-16)).sum()
            e = yest  # note: can copy if yest needed later
            e[bi, yi] -= 1
            grad = np.dot(a.T, e)
            if lamb > 0:
                cost += 0.5 * lamb * (X**2).sum()
                grad += lamb * X
            return cost, grad

        X = rng.normal(scale=1./m, size=Xshape) if X is None else X.copy()
        V = np.zeros_like(X)
        for i_epoch in range(self.n_epochs):
            epoch_cost = 0
            for a, yi in batches():
                mu = abs(momentum)
                if mu > 0:
                    V *= mu
                X2 = X - eta*V if momentum < 0 else X
                cost, grad = f_df(a, X2, yi)
                epoch_cost += cost
                if mu > 0:
                    V += grad
                    X -= eta * V
                else:
                    X -= eta * grad

            if self.verbose >= 1:
                print("Epoch %3d: %0.2e" % (i_epoch, epoch_cost))

        t = time.time() - tstart

        return X, {
            'rmses': npext.rms(softmax(np.dot(A, X), axis=1) - Y, axis=1),
            'time': t,
            'iterations': self.n_epochs,
        }


class HingeLoss(nengo.solvers.Solver):
    """Solver to minimize the hinge loss for a categorization output.

    TODO
    """

    reg = NumberParam('reg', low=0)
    n_epochs = IntParam('n_epochs', low=1)
    verbose = IntParam('verbose')

    def __init__(self, reg=0, weights=False, verbose=0):
        super(HingeLoss, self).__init__(weights=weights)
        self.reg = reg
        self.verbose = verbose
        self.n_epochs = 1000

    def __call__(self, A, Y, rng=None, E=None):
        import scipy.optimize

        tstart = time.time()

        assert A.shape[0] == Y.shape[0]
        m, n = A.shape
        _, d = Y.shape
        Xshape = (n, d)

        # regularization
        sigma = self.reg * A.max()
        lamb = m * sigma**2

        # --- initialization
        X0 = rng.uniform(-1./n, 1./n, size=Xshape)
        # X0, _ = nengo.solvers.LstsqL2(reg=self.reg)(A, Y, rng=rng, E=E)  # a little better

        # --- solve with L-BFGS
        yi = Y.argmax(axis=1)

        def f_df(x):
            X = x.reshape(Xshape)
            Z = np.dot(A, X)

            # Crammer and Singer (2001) version
            zy = Z[np.arange(m), yi]
            Z[np.arange(m), yi] = -np.inf
            ti = Z.argmax(axis=1)
            zt = Z[np.arange(m), ti]
            margins = zy - zt

            E = np.zeros(Z.shape)
            margin1 = margins < 1
            E[margin1, yi[margin1]] = -1
            E[margin1, ti[margin1]] = 1

            cost = np.maximum(0, 1 - margins).sum()
            grad = np.dot(A.T, E)
            if lamb > 0:
                cost += 0.5 * lamb * (X**2).sum()
                grad += lamb * X
            return cost, grad.ravel()

        w0 = X0.ravel()
        w, mincost, info = scipy.optimize.fmin_l_bfgs_b(
            f_df, w0, maxfun=self.n_epochs, iprint=self.verbose)

        t = time.time() - tstart

        X = w.reshape(Xshape)
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': t,
            'iterations': info['funcalls'],
        }


class LDA_OVA(nengo.solvers.Solver):
    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        assert E is None
        assert A.ndim == Y.ndim == 2 and A.shape[0] == Y.shape[0]

        m, n = A.shape
        d = Y.shape[1]
        y = np.argmax(Y, axis=1)

        cov_c = lambda Ac: np.dot(Ac.T, Ac) / (Ac.shape[0] - 1)
        cov = lambda Ai: cov_c(Ai - Ai.mean(axis=0))
        mu_i = np.array([A[y == i].mean(axis=0) for i in range(d)])
        mu_j = np.array([A[y != i].mean(axis=0) for i in range(d)])
        Sigma_i = [cov(A[y == i]) for i in range(d)]
        Sigma_j = [cov(A[y != i]) for i in range(d)]

        X = np.zeros((n, d))
        for i in range(d):
            Sigma = 0.5 * (Sigma_i[i] + Sigma_j[i])
            x = cho_solve(Sigma, mu_i[i] - mu_j[i])
            X[:, i] = x
            # NOTE: we don't solve for threshold, because we can't control
            #  the bias on output. Hopefully this doesn't matter because we're
            #  comparing between classes.

        return X, {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': time.time() - tstart}


class LstsqClassifier(nengo.solvers.Solver):
    """Solve for weights for one-hot classification.

    Uses weighted least squares to solve for better classification weights.

    Parameters
    ----------
    weights : bool, optional (Default: False)
        If False, solve for decoders. If True, solve for weights.
    reg : float, optional (Default: 0.1)
        Amount of L2 regularization, as a fraction of the neuron activity.
    weight_power : float, optional (Default: 1)
        Exponent for the weights.

    Attributes
    ----------
    precompute_ai : bool (Default: True)
        Whether to precompute the subcomponents of the Gram matrix. Much faster
        computation at the expense of slightly more memory.
    """

    reg = NumberParam('reg', low=0)
    weight_power = NumberParam('weight_power', low=0)
    precompute_ai = BoolParam('precompute_ai')

    def __init__(self, weights=False, reg=0.1, weight_power=1):
        super(LstsqClassifier, self).__init__(weights=weights)
        self.reg = reg
        self.weight_power = weight_power
        self.precompute_ai = True

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()

        m, n = A.shape
        _, d = Y.shape
        sigma = self.reg * A.max()
        precompute_ai = self.precompute_ai

        Y = Y > 0.5  # ensure Y is binary

        def getAAi(i, y, cache={}):
            if i in cache:
                return cache[i]

            Ai = A[y]
            AAi = np.dot(Ai.T, Ai)
            if precompute_ai:
                cache[i] = AAi
            return AAi

        if not precompute_ai:
            AA = np.dot(A.T, A)
        else:
            AA = np.zeros((n, n))
            for i in range(d):
                AA += getAAi(i, Y[:, i])

        X = np.zeros((n, d))
        for i in range(d):
            y = Y[:, i]

            # weight for classification
            p = y.mean()
            q = self.weight_power
            wr = p*(1-p)**q + (1-p)*p**q
            w0 = p**q / wr
            w1 = (1-p)**q / wr
            dw = w1 - w0
            w = w0 + dw*y

            # form Gram matrix G = A.T W A + m * sigma**2
            G = w0*AA + dw*getAAi(i, y)
            np.fill_diagonal(G, G.diagonal() + m * sigma**2)
            b = np.dot(A.T, w * y)

            X[:, i] = cho_solve(G, b, overwrite=True)

        tend = time.time()
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': tend - tstart}


def classifier_weighted_lstsq(A, yi, d, sigma, precompute_ai=True, weight_power=1):
    m, n = A.shape

    def getAAi(i, cache={}):
        if i in cache:
            return cache[i]

        Ai = A[yi == i]
        AAi = np.dot(Ai.T, Ai)
        if precompute_ai:
            cache[i] = AAi
        return AAi

    if not precompute_ai:
        AA = np.dot(A.T, A)
    else:
        AA = np.zeros((n, n))
        for i in range(d):
            AA += getAAi(i)

    X = np.zeros((n, d))
    for i in range(d):
        y = (yi == i).astype(np.float64)

        # weight for classification
        p = y.mean()
        q = weight_power
        wr = p*(1-p)**q + (1-p)*p**q
        w0 = p**q / wr
        w1 = (1-p)**q / wr
        dw = w1 - w0
        w = w0 + dw*y

        # form Gram matrix G = A.T W A + m * sigma**2
        G = w0*AA + dw*getAAi(i)
        np.fill_diagonal(G, G.diagonal() + m * sigma**2)
        b = np.dot(A.T, w * y)

        X[:, i] = cho_solve(G, b, overwrite=True)

    return X


class LstsqClassifierParts(nengo.solvers.Solver):
    """Have N independent classifiers, each for part of the neurons, combine.
    """

    reg = NumberParam('reg', low=0)
    weight_power = NumberParam('weight_power', low=0)
    precompute_ai = BoolParam('precompute_ai')

    def __init__(self, weights=False, reg=0.1):
        super(LstsqClassifierParts, self).__init__(weights=weights)
        self.reg = reg
        self.weight_power = 1.
        self.precompute_ai = True

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()

        m, n = A.shape
        _, d = Y.shape
        yi = Y.argmax(axis=1)

        sigma = self.reg * A.max()
        precompute_ai = self.precompute_ai
        weight_power = self.weight_power

        X = np.zeros((n, d))

        blocks = 10
        nb, rb = n // blocks, n % blocks
        nblock = [nb + (1 if i < rb else 0) for i in range(blocks)]
        i, j = 0, 0
        for k in range(blocks):
            i, j = j, j + nblock[k]
            X[i:j] = classifier_weighted_lstsq(
                A[:, i:j], yi, d, sigma,
                weight_power=weight_power, precompute_ai=precompute_ai)

        tend = time.time()
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': tend - tstart}


# class LstsqClassifierAda(nengo.solvers.Solver):
#     """TODO
#     """

#     reg = NumberParam('reg', low=0)
#     weight_power = NumberParam('weight_power', low=0)
#     precompute_ai = BoolParam('precompute_ai')

#     def __init__(self, weights=False, reg=0.1):
#         super(LstsqClassifierParts, self).__init__(weights=weights)
#         self.reg = reg
#         self.weight_power = 1.
#         self.precompute_ai = True

#     def __call__(self, A, Y, rng=None, E=None):
#         tstart = time.time()

#         m, n = A.shape
#         _, d = Y.shape
#         yi = Y.argmax(axis=1)

#         sigma = self.reg * A.max()
#         precompute_ai = self.precompute_ai
#         # weight_power = self.weight_power

#         X = np.zeros((n, d))

#         if 1:
#             W = (1./m) * np.ones((m, d))
#         # else:
#         #     q = weight_power
#         #     wr = p*(1-p)**q + (1-p)*p**q
#         #     w0 = p**q / wr
#         #     w1 = (1-p)**q / wr
#         #     dw = w1 - w0
#         #     w = w0 + dw*y


#         blocks = 10
#         nb, rb = n // blocks, n % blocks
#         nblock = [nb + (1 if i < rb else 0) for i in range(blocks)]
#         i, j = 0, 0
#         for k in range(blocks):
#             i, j = j, j + nblock[k]

#             # compute new classifier

#             X[i:j] = classifier_weighted_lstsq(
#                 A[:, i:j], yi, d, sigma,
#                 weight_power=weight_power, precompute_ai=precompute_ai)

#         tend = time.time()
#         return self.mul_encoders(X, E), {
#             'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
#             'time': tend - tstart}


class LstsqLOO(nengo.solvers.Solver):
    """Use Rifkin's method for finding optimal leave-one-out regularization.

    References
    ----------
    Rifkin, R. M., & Lippert, R. A. (2007). Notes on regularized least squares.
        Massachusetts Institute of Technology.
    """

    def __init__(self, weights=False):
        super(LstsqLOO, self).__init__(weights=weights)

    def __call__(self, A, Y, rng=None, E=None):
        try:
            from scipy.optimize import fminbound
        except ImportError:
            fminbound = None

        tstart = time.time()

        # m, n = A.shape
        # _, d = Y.shape
        m = A.shape[0]

        Amax = A.max()
        get_lambda = lambda reg: m * (reg * Amax)**2

        AA = np.dot(A.T, A)
        e, Q = np.linalg.eigh(AA)
        P = np.dot(A, Q)
        PY = np.dot(P.T, Y)
        P2 = P**2

        def looe(reg):  # leave-one-out error for given regularization
            lamb = get_lambda(reg)
            c = Y - np.dot(P, PY / (e[:, None] + lamb))
            d = (P2 / (e + lamb)).sum(axis=1)
            return ((c / (1 - d)[:, None])**2).sum()

        # find regularization that minimizes LOOE
        b0, b1 = 10.**(-3), 0.2
        if fminbound is None:  # do a simple grid search
            regs = np.logspace(np.log10(b0), np.log10(b1), 25)
            looes = [looe(reg) for reg in regs]
            reg = regs[np.argmin(looes)]
        else:  # assume unimodal function (only one local minimum) and search
            reg = fminbound(looe, b0, b1, xtol=5e-4, maxfun=50)

        lamb = get_lambda(reg)
        X = np.dot(Q, PY / (e[:, None] + lamb))

        tend = time.time()

        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': tend - tstart,
            'reg': reg}


class LstsqClassifierLOO(nengo.solvers.Solver):
    """Least squares, weighted for classification, with optimal leave-one-out.

    This method combines weighted least-squares for classification
    with Rifkin's method for finding optimal leave-one-out regularization.

    References
    ----------
    Rifkin, R. M., & Lippert, R. A. (2007). Notes on regularized least squares.
        Massachusetts Institute of Technology.
    """

    precompute_ai = BoolParam('precompute_ai')

    def __init__(self, weights=False):
        super(LstsqClassifierLOO, self).__init__(weights=weights)
        self.precompute_ai = True

    def __call__(self, A, Y, rng=None, E=None):
        try:
            from scipy.optimize import fminbound
        except ImportError:
            fminbound = None

        tstart = time.time()

        m, n = A.shape
        _, d = Y.shape

        precompute_ai = self.precompute_ai
        Y = Y > 0.5  # ensure Y is binary

        Amax = A.max()
        get_lambda = lambda reg: m * (reg * Amax)**2

        def getAAi(i, y, cache={}):
            if i in cache:
                return cache[i]

            Ai = A[y]
            AAi = np.dot(Ai.T, Ai)
            if precompute_ai:
                cache[i] = AAi
            return AAi

        if not precompute_ai:
            AA = np.dot(A.T, A)
        else:
            AA = np.zeros((n, n))
            for i in range(d):
                AA += getAAi(i, Y[:, i])

        # easy method: compute best lambda individually for each weighting
        X = np.zeros((n, d))
        regs = []
        for i in range(d):
            y = Y[:, i]

            # weight for classification
            p = y.mean()
            w0 = 0.5 / (1-p)
            w1 = 0.5 / p
            dw = w1 - w0
            w = w0 + dw*y

            # form Gram matrix G = A.T W A + m * sigma**2
            G = w0*AA + dw*getAAi(i, y)
            e, Q = np.linalg.eigh(G)
            P = np.dot(A, Q)
            PY = np.dot(P.T, w * y)

            def looe(reg):  # leave-one-out error for given regularization
                lamb = get_lambda(reg)
                c = y - np.dot(P, PY / (e + lamb))
                d = (P**2 / (e + lamb)).sum(axis=1)
                return ((c / (1 - d))**2).sum()

            # find regularization that minimizes LOOE
            b0, b1 = 10.**(-3), 0.2
            if fminbound is None:  # do a simple grid search
                regspace = np.logspace(np.log10(b0), np.log10(b1), 31)
                looes = [looe(reg) for reg in regspace]
                reg = regspace[np.argmin(looes)]
            else:  # assume unimodal function (only one minimum) and search
                reg = fminbound(looe, b0, b1, xtol=5e-4, maxfun=50)

            lamb = get_lambda(reg)
            X[:, i] = np.dot(Q, PY / (e + lamb))
            regs.append(reg)


        # other method: compute single lambda for all systems

        # --- eigendecompositions for all weightings
        # store weight vectors as well

        # --- for each weighting and lamb, compute c and d coefficients via P = AQ
        # - this will be expensive, because we now have D gemms of O(MN^2)
        # - the weighting should appear somewhere else other than just in Q,
        #   maybe just on Y

        tend = time.time()

        print(regs)

        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': tend - tstart,
            'regs': regs}

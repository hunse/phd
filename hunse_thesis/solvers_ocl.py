import time

import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.reduction

import nengo.utils.numpy as npext

import nengo_ocl

from hunse_thesis.solvers import Softmax, HingeLoss, LstsqClassifier


def plan_softmax(queue, X, Y):
    from mako.template import Template
    from nengo_ocl.utils import as_ascii
    from nengo_ocl.plan import Plan

    m, n = X.shape
    assert n <= 32
    assert Y.shape == X.shape
    assert X.elemstrides[1] == 1
    assert Y.elemstrides[1] == 1

    text = """
        __kernel void fn(
            __global const ${Xtype} *X,
            __global ${Ytype} *Y
        )
        {
            const int i = get_global_id(0);

            ${Xtype} ex[${n}];
            __global const ${Xtype} *x = X + i*${Xstride0};
            __global ${Ytype} *y = Y + i*${Ystride0};

            ${Xtype} maxx = -INFINITY;
            for (int j = 0; j < ${n}; j++)
                if (x[j] > maxx)
                    maxx = x[j];

            ${Xtype} sumex = 0;
            for (int j = 0; j < ${n}; j++) {
                ex[j] = exp(x[j] - maxx);
                sumex += ex[j];
            }

            for (int j = 0; j < ${n}; j++)
                y[j] = ex[j] / sumex;
        }
        """
    textconf = dict(Xtype=X.ctype, Ytype=Y.ctype, m=m, n=n,
                    Xstride0=X.elemstrides[0], Ystride0=Y.elemstrides[0])
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    fn = cl.Program(queue.context, text).build().fn
    fn.set_args(*[arr.data for arr in (X, Y)])
    plan = Plan(queue, fn, gsize=(m,))
    return plan


@nengo_ocl.builder.Builder.register_ocl_solver(Softmax)
def solve_Softmax(solver, queue, clA, Y, rng=None, E=None):
    from nengo_extras.convnet import softmax
    import scipy.optimize
    import pyopencl_blas
    pyopencl_blas.setup()

    tstart = time.time()

    assert clA.shape[0] == Y.shape[0]
    m, n = clA.shape
    _, d = Y.shape
    Xshape = (n, d)

    # regularization
    sigma = solver.reg * cl.array.max(clA).get()
    lamb = m * sigma**2

    # --- initialization
    # X0 = np.zeros(Xshape, dtype=np.float32)
    X0 = np.zeros(Xshape, dtype=np.float64)

    # --- solve with L-BFGS
    clY = cl.array.to_device(queue, Y.astype(np.float32))
    clyi = cl.array.to_device(queue, np.argmax(Y, axis=1).astype(np.int32))
    clX = cl.array.Array(queue, (n, d), dtype=np.float32)
    clE = cl.array.Array(queue, (m, d), dtype=np.float32)
    clG = cl.array.Array(queue, (n, d), dtype=np.float32)

    softmax_plan = plan_softmax(queue, clE, clE)

    # sum_square = cl.reduction.ReductionKernel(
    #     queue.context, np.float32, neutral="0",
    #     reduce_expr="a+b", map_expr="x[i]*x[i]",
    #     arguments="__global float *x")

    sum_logloss = cl.reduction.ReductionKernel(
        queue.context, np.float32, neutral="0", reduce_expr="a+b",
        map_expr="-log(max(Y[i*%(d)d + yi[i]], 1e-16f))" % dict(d=d),
        arguments="__global const int *yi, __global const float *Y")
    assert clE.elemstrides[0] == d

    def f_df(x):
        clX.set(x.astype(np.float32).reshape(Xshape))
        pyopencl_blas.gemm(queue, clA, clX, clE)
        softmax_plan()
        cost = sum_logloss(clyi, clE).get()
        clE[:] -= clY
        pyopencl_blas.gemm(queue, clA, clE, clG, transA=True)
        if lamb > 0:
            cost += 0.5 * lamb * pyopencl.array.sum(clX**2).get()
            # cost += 0.5 * lamb * sum_square(clX).get()
            clG[:] += lamb * clX

        G = clG.get().astype(np.float64)
        return cost, G.ravel()

    x0 = X0.ravel()
    x, mincost, info = scipy.optimize.fmin_l_bfgs_b(
        f_df, x0, maxfun=solver.n_epochs, iprint=solver.verbose)

    tend = time.time()

    A = clA.get()
    X = x.reshape(Xshape)
    return solver.mul_encoders(X, E), {
        'rmses': npext.rms(softmax(np.dot(A, X), axis=1) - Y, axis=1),
        'time': tend - tstart}


def plan_hingeloss(queue, yinds, Z, c, E):
    from mako.template import Template
    from nengo_ocl.utils import as_ascii
    from nengo_ocl.plan import Plan

    m, n = Z.shape
    assert n <= 32
    assert Z.shape == E.shape
    assert Z.elemstrides[1] == 1
    assert E.elemstrides[1] == 1
    assert yinds.shape == (m,)
    assert yinds.elemstrides[0] == 1
    assert c.shape == (m,)
    assert c.elemstrides[0] == 1

    text = """
        __kernel void fn(
            __global const ${yindstype} *yinds,
            __global const ${Ztype} *Z,
            __global ${ctype} *c,
            __global ${Etype} *E
        )
        {
            const int i = get_global_id(0);

            const ${yindstype} yi = yinds[i];
            __global const ${Ztype} *z = Z + i*${Zstride0};
            __global ${Etype} *e = E + i*${Estride0};

            ${yindstype} ti;
            ${Ztype} zj, zy, zt = -INFINITY;
            zt = -INFINITY;
            for (int j = 0; j < ${n}; j++) {
                e[j] = 0;
                zj = z[j];
                if (j == yi) {
                    zy = zj;
                } else if (zj > zt) {
                    zt = zj;
                    ti = j;
                }
            }

            ${Ztype} margin = zy - zt;
            if (margin < 1) {
                e[yi] = -1;
                e[ti] = 1;
            }
            c[i] = max(1 - margin, 0.0f);
        }
        """
    textconf = dict(yindstype=yinds.ctype, Ztype=Z.ctype, ctype=c.ctype,
                    Etype=E.ctype, m=m, n=n,
                    Zstride0=Z.elemstrides[0], Estride0=E.elemstrides[0])
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    fn = cl.Program(queue.context, text).build().fn
    fn.set_args(*[arr.data for arr in (yinds, Z, c, E)])
    plan = Plan(queue, fn, gsize=(m,))
    return plan


@nengo_ocl.builder.Builder.register_ocl_solver(HingeLoss)
def solve_HingeLoss(solver, queue, clA, Y, rng=None, E=None):
    import scipy.optimize
    import pyopencl_blas
    pyopencl_blas.setup()

    tstart = time.time()

    assert clA.shape[0] == Y.shape[0]
    m, n = clA.shape
    _, d = Y.shape
    Xshape = (n, d)

    # regularization
    sigma = solver.reg * cl.array.max(clA).get()
    lamb = m * sigma**2

    # --- initialization
    X0 = rng.uniform(-1./n, 1./n, size=Xshape)

    # --- solve with L-BFGS
    yinds = Y.argmax(axis=1)

    clX = cl.array.Array(queue, (n, d), dtype=np.float32)
    clyinds = cl.array.to_device(queue, yinds.astype(np.int32))
    clZ = cl.array.Array(queue, (m, d), dtype=np.float32)
    clc = cl.array.Array(queue, (m,), dtype=np.float32)
    clE = cl.array.Array(queue, (m, d), dtype=np.float32)
    clG = cl.array.Array(queue, (n, d), dtype=np.float32)

    hingeloss_plan = plan_hingeloss(queue, clyinds, clZ, clc, clE)

    def f_df(x):
        clX.set(x.astype(np.float32).reshape(Xshape))
        pyopencl_blas.gemm(queue, clA, clX, clZ)
        hingeloss_plan()

        cost = pyopencl.array.sum(clc).get()
        pyopencl_blas.gemm(queue, clA, clE, clG, transA=True)
        if lamb > 0:
            cost += 0.5 * lamb * pyopencl.array.sum(clX**2).get()
            # cost += 0.5 * lamb * sum_square(clX).get()
            clG[:] += lamb * clX

        G = clG.get().astype(np.float64)
        return cost, G.ravel()

    x0 = X0.ravel()
    x, mincost, info = scipy.optimize.fmin_l_bfgs_b(
        f_df, x0, maxfun=solver.n_epochs, iprint=solver.verbose)

    tend = time.time()

    A = clA.get()
    X = x.reshape(Xshape)
    return solver.mul_encoders(X, E), {
        'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
        'time': tend - tstart}


@nengo_ocl.builder.Builder.register(LstsqClassifier)
def build_lstsqclassifier(model, solver, conn, rng, transform):
    from nengo.builder.connection import multiply
    from nengo_ocl.builder.solvers import (
        get_solve_params, wrap_solver, solve_for_decoders)
    eval_points, neuron_type, gain, bias, X, Y, E = get_solve_params(
        model, solver, conn, rng, transform)

    # sort eval points by category
    assert Y.ndim == 2
    Yi = np.argmax(Y, axis=1)
    i = np.argsort(Yi)
    X[:] = X[i]
    Y[:] = Y[i]

    wrapped_solver = wrap_solver(model, conn, solve_for_decoders)
    decoders, solver_info = wrapped_solver(
        solver, neuron_type, gain, bias, X, Y, rng,
        E=E, conn=conn, queue=model.builder.queue)
    weights = (decoders.T if solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


@nengo_ocl.builder.Builder.register_ocl_solver(LstsqClassifier)
def solve_lstsqclassifier(solver, queue, clA, Y, rng=None, E=None):
    # from nengo_ocl.builder.solvers import cho_solve
    import pyopencl_blas
    pyopencl_blas.setup()

    m, n = clA.shape
    _, d = Y.shape
    precompute_ai = solver.precompute_ai


    def XTdotX(clX):
        clXX = cl.array.Array(queue, (n, n), dtype=np.float32)
        pyopencl_blas.gemm(queue, clX, clX, clXX, transA=True)
        return clXX.get()

    def ATdotx(x):
        clx = cl.array.to_device(queue, x.astype(np.float32))
        cly = cl.array.Array(queue, (n,), dtype=np.float32)
        pyopencl_blas.gemv(queue, clA, clx, cly, transA=True)
        return cly.get()

    def AdotX(X):
        clX = cl.array.to_device(queue, X.astype(np.float32))
        clAX = cl.array.Array(queue, (m, clX.shape[1]), dtype=np.float32)
        pyopencl_blas.gemm(queue, clA, clX, clAX)
        return clAX.get()

    def getAi(i, cache={}):
        if i in cache:
            return cache[i]

        clAi = clAis[i]
        AAi = XTdotX(clAi)
        if precompute_ai:
            cache[i] = AAi
        return AAi


    tstart = time.time()

    sigma = solver.reg * cl.array.max(clA).get()

    # Get Y inds
    Yi = np.argmax(Y, axis=1)
    Yd = np.diff(Yi)
    assert set(np.unique(Yd)) == set((0, 1)), "Y not sorted, or missing some classes"

    clAis = []
    for i in range(d):
        inds, = (Yi == i).nonzero()
        a, b = inds.min(), inds.max()+1
        clAis.append(clA[a:b])

    if not precompute_ai:
        AA = XTdotX(clA)
    else:
        AA = np.zeros((n, n))
        for i in range(d):
            AA += getAi(i)

    X = np.zeros((n, d))
    for i in range(d):
        y = Y[:, i]

        # weight for classification
        p = y.mean()
        q = solver.weight_power
        wr = p*(1-p)**q + (1-p)*p**q
        w0 = p**q / wr
        w1 = (1-p)**q / wr
        dw = w1 - w0
        w = w0 + dw*y

        # form Gram matrix G = A.T W A + m * sigma**2
        G = w0*AA + dw*getAi(i)
        np.fill_diagonal(G, G.diagonal() + m * sigma**2)
        b = ATdotx(w * y)

        # X[:, i] = cho_solve(G, b, overwrite=True)
        X[:, i] = np.linalg.solve(G, b)

    tend = time.time()

    AX = AdotX(X)
    return solver.mul_encoders(X, E), {
        'rmses': npext.rms(AX - Y, axis=1),
        'time': tend - tstart}

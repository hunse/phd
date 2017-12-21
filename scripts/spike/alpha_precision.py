"""
Test that softlifalpha approaches real rate as synapse time constant approaches infinity
"""
import matplotlib.pyplot as plt
import numpy as np

# p = np.float32(3.)
# tau = np.float32(100000.)

# rho = np.exp(-p/tau)

# rho1 = (1 - rho) * tau

# print(rho1)


def compute_series(p, u, tau, dtype=np.float64):
    p = np.asarray(p, dtype=dtype)
    u = np.asarray(u, dtype=dtype)
    tau = np.asarray(tau, dtype=dtype)

    p = p[:, None, None]
    u = u[None, :, None]
    tau = tau[None, None, :]
    t = p * u

    # --- arithmetico-geometric series
    a = t
    b = (1./tau**2) * np.exp(-t/tau)
    d = p
    r = np.exp(-p/tau)
    s = b * (a / (1 - r) + d*r / (1 - r)**2)

    return s


def compute_series2(p, u, tau, dtype=np.float64):
    p = np.asarray(p, dtype=dtype)
    u = np.asarray(u, dtype=dtype)
    tau = np.asarray(tau, dtype=dtype)

    p = p[:, None, None]
    u = u[None, :, None]
    tau = tau[None, None, :]
    t = p * u

    q_s = np.exp(-t/tau)
    r_s1 = -np.expm1(-p/tau)
    rhot = tau * r_s1
    pt = np.where(p < 100*tau, (p - t)*(1 - r_s1), 0.)
    qt = np.where(t < 100*tau, q_s * (t + pt), 0.)
    r = qt / (rhot*rhot)

    return r


def test_rho():
    taus = np.logspace(-10, 10, 101)
    p = 3.
    # p = np.float32(1.)

    p32 = np.float32(p)

    def f0(taus):
        # return taus * (1 - np.exp(-p / taus))
        return -taus * np.expm1(-p / taus)

    def f1(taus):
        taus = taus.astype(np.float32)
        return taus * (1 - np.exp(-p32 / taus))

    def f2(taus):
        taus = taus.astype(np.float32)
        rho = taus * (1 - np.exp(-p32 / taus))
        rho[taus > 1e5] = p32
        return rho

    def f3(taus):
        taus = taus.astype(np.float32)
        rho = taus * (1 - np.exp(-p32 / taus))
        rho[taus > 1e4] = p32
        return rho

    def f4(taus):
        taus = taus.astype(np.float32)
        return -taus * np.expm1(-p32 / taus)

    fs = [f0, f1, f2, f3, f4]
    ys = [f(taus) for f in fs]
    es = [(y - ys[0]) for y in ys]

    plt.figure()
    plt.subplot(311)
    for y in ys:
        plt.semilogx(taus, y)

    plt.subplot(312)
    for e in es:
        plt.semilogx(taus, e)

    plt.subplot(313)
    for e in es:
        plt.loglog(taus, np.abs(e) + 1e-20)


def test_period():
    # ps = np.logspace(-5, 4, 51)
    # ps = np.logspace(-5, 5, 51)
    # ps = np.logspace(-5, 10, 51)
    # ps = np.logspace(-5, 50, 51)

    maxe = np.floor(np.log10(np.nextafter(np.array([np.inf], dtype=np.float32), 0))).astype(np.float64)
    # print(maxe)
    ps = np.logspace(-5, maxe, 11)

    # us = np.array([0, 0.2])
    # us = np.linspace(0, 1, 6)
    us = (1./5) * np.arange(5)
    tau = np.array([0.003])
    # tau = np.array([1000.])

    y0 = compute_series(ps, us, tau)[:, :, 0]
    # y1 = compute_series(ps, us, tau, dtype=np.float16)[:, :, 0]
    # y1 = compute_series(ps, us, tau, dtype=np.float32)[:, :, 0]
    # y1 = compute_series2(ps, us, tau, dtype=np.float16)[:, :, 0]
    y1 = compute_series2(ps, us, tau, dtype=np.float32)[:, :, 0]

    print("All finite: %s" % np.isfinite(y1).all())
    # print(y1)

    print(y0[-1])
    print(y1[-1])

    plt.figure()
    plt.subplot(211)
    plt.loglog(ps, y0)
    plt.ylim([1e-32, None])

    plt.subplot(212)
    plt.loglog(ps, np.abs(y0 - y1))


# def test_long_perio



if __name__ == '__main__':
    test_rho()
    test_period()

    plt.show()

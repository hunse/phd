
import numpy as np
# import quantities as q
import matplotlib.pyplot as plt
import scipy.integrate

import seaborn as sns
sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')


def ndyn(V):
    alpha_n = 0.01*(V + 55) / (1 - np.exp(-0.1*(V + 55)))
    alpha_n[np.isnan(alpha_n)] = 0.1
    beta_n = 0.125*np.exp(-0.0125*(V + 65))
    tau_n = 1. / (alpha_n + beta_n)
    n_inf = alpha_n * tau_n
    return tau_n, n_inf


def mdyn(V):
    alpha_m = 0.1*(V + 40) / (1 - np.exp(-0.1*(V + 40)))
    alpha_m[np.isnan(alpha_m)] = 1.0
    beta_m = 4*np.exp(-0.0556*(V + 65))
    tau_m = 1. / (alpha_m + beta_m)
    m_inf = alpha_m * tau_m
    return tau_m, m_inf


def hdyn(V):
    alpha_h = 0.07*np.exp(-0.05*(V + 65))
    beta_h = 1. / (1 + np.exp(-0.1*(V + 35)))
    tau_h = 1. / (alpha_h + beta_h)
    h_inf = alpha_h * tau_h
    return tau_h, h_inf


def hh_sim(dt, t_final, fu, state, N=1,
           gbar_L=0.003, gbar_K=0.36, gbar_Na=1.2,
           E_L=-54.387, E_K=-77, E_Na=50, c_m=0.01):

    def I_L(y):
        v, n, m, h = y
        return gbar_L*(E_L - v)

    def I_Na(y):
        v, n, m, h = y
        return gbar_Na*m**3*h*(E_Na - v)

    def I_K(y):
        v, n, m, h = y
        return gbar_K*n**4*(E_K - v)

    def sim_function(t, y):
        y = y.reshape((4, -1))
        v, n, m, h = y

        I = fu(t) + I_L(y) + I_Na(y) + I_K(y)
        dv = I/c_m
        tau_n, n_inf = ndyn(v)
        tau_m, m_inf = mdyn(v)
        tau_h, h_inf = hdyn(v)
        dn = (n_inf - n)/tau_n
        dm = (m_inf - m)/tau_m
        dh = (h_inf - h)/tau_h
        return np.array([dv, dn, dm, dh]).ravel()

    state = np.asarray(state)
    if state.ndim == 1:
        state = np.column_stack([s*np.ones(N) for s in state])

    r = scipy.integrate.ode(sim_function).set_integrator('dopri5')
    r.set_initial_value(state.ravel())

    t = []
    u = []
    states = []
    currents = []
    while r.successful() and r.t < t_final:
        r.integrate(r.t + dt)
        t.append(r.t)
        u.append(fu(r.t))

        y = r.y.reshape(4, -1)
        states.append(y)
        currents.append((I_L(y), I_Na(y), I_K(y)))

    t = np.array(t)
    u = np.array(u)
    states = np.array(states)
    currents = np.array(currents)

    return t, u, states, currents


dayan_params = dict(gbar_L=0.003, gbar_K=0.36, gbar_Na=1.2,
                    E_L=-54.387, E_K=-77, E_Na=50)

# u = 0.2
# u = 0.01
# u = lambda t: ((t > 10) & (t < 12)) * 0.04
fu = lambda t: 0.08 * max(1 - abs(t - 2), 0)

# u = lambda t: 0.0
# dt = 0.2
dt = 0.01
t_final = 15.
# t_final = 100.

N = 1
# s0 = [-64.2, 0.330, 0.058, 0.568]
s0 = [-65.0, 0.317, 0.053, 0.596]
# s0 = [-70, 0, 0, 0]
# s0 = [-65, 0, 0, 0]

t, u, states, currents = hh_sim(dt, t_final, fu, s0, N=N, **dayan_params)

fig = plt.figure(figsize=(4, 5))
ax0 = plt.subplot2grid((5, 1), (0, 0))
ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=2)
ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

ax0.plot(t, u)
ax0.set_xticklabels([])
ax0.set_yticks(np.linspace(0, 0.08, 5))
ax0.set_ylabel('input current [$\mu$A]')

ax1.plot(t, states[:, 0, 0])
ax1.set_xticklabels([])
ax1.set_ylabel('membrane voltage [mV]')

# ax2.plot(t, u, label='external')
ax2.plot(t, currents[:, 1, 0], label='Na')
ax2.plot(t, currents[:, 2, 0], label='K')
ax2.plot(t, currents[:, 1, 0] + currents[:, 2, 0], label='Na + K')
ax2.set_xlabel('time [ms]')
ax2.set_ylabel('current [$\mu$A]')
ax2.legend(loc='best')

plt.tight_layout()
plt.savefig('hh_spike.pdf')


# plt.figure()
# plt.plot(t, states[:, 1:, 0])
# plt.legend(('n', 'm', 'h'), loc='best')


plt.show()

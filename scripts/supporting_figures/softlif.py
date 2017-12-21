import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from hunse_thesis.neurons import lif, dlif, softlif, dsoftlif

sns.set_style('white')
sns.set(context='paper', style='ticks', palette='dark')


# sigma = 0.2
# sigma = 1.0
sigma = 0.01
tau_ref = 0.002
tau_rc = 0.03


x = np.linspace(0.5, 1.5, 10001).astype('float64')

y = lif(x, tau_rc, tau_ref)
dy = dlif(x, tau_rc, tau_ref)
z = softlif(x, tau_rc, tau_ref, sigma=sigma)
dz = dsoftlif(x, tau_rc, tau_ref, sigma=sigma)

plt.figure(figsize=(6.4, 3.5))

plt.subplot(121)
plt.plot(x, y, 'k')
plt.plot(x, z, 'k--')
# plt.xlim([-0.5, 0.5])
plt.xlim([0.5, 1.5])
plt.ylim([-1.5, 30])
plt.xlabel('input current ($j$)')
plt.ylabel('firing rate ($r$) [Hz]')
plt.legend(['LIF', 'soft LIF'], loc=2, )

plt.subplot(122)
plt.plot(x, dy, 'k')
plt.plot(x, dz, 'k--')
# plt.xlim([-0.5, 0.5])
plt.xlim([0.5, 1.5])
plt.ylim([-10, 300])
plt.xlabel('input current ($j$)')
plt.ylabel('firing rate derivative ($dr / dj$)')

plt.tight_layout()
plt.savefig('softlif.pdf')

plt.show()

# https://media.readthedocs.org/pdf/pyhht/latest/pyhht.pdf

from scipy.signal import hilbert
from scipy import angle, unwrap
import numpy as np
import matplotlib.pyplot as plt

plt.grid(True)

x = np.linspace(0, 2 * np.pi, 1000)
s1 = np.sin(x)
s2 = np.sin(x) - 1
s3 = np.sin(x) + 2
# plt.plot(x, s1, 'b', x, s2, 'g', x, s3, 'r')

hs1 = hilbert(s1)
hs2 = hilbert(s2)
hs3 = hilbert(s3)
# plt.plot(np.real(hs1), np.imag(hs1), 'b')
# plt.plot(np.real(hs2), np.imag(hs2), 'g')
# plt.plot(np.real(hs3), np.imag(hs3), 'r')

omega_s1 = unwrap(angle(hs1))  # unwrapped instantaneous phase
omega_s2 = unwrap(angle(hs2))
omega_s3 = unwrap(angle(hs3))
f_inst_s1 = np.diff(omega_s1)  # instantaneous frequency
f_inst_s2 = np.diff(omega_s2)
f_inst_s3 = np.diff(omega_s3)
# plt.plot(x[1:], f_inst_s1, "b")
# plt.plot(x[1:], f_inst_s2, "g")
plt.plot(x[1:], f_inst_s3, "r")
plt.show()

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
# atili: addition imports
import scipy.io.wavfile

signalType = 1

# In this example we use the Hilbert transform to determine the amplitude envelope and instantaneous frequency
# of an amplitude-modulated signal.

# Note that End-point limitation exists in the example

if 0 == signalType:
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    # We create a chirp of which the frequency increases from 20 Hz to 100 Hz and apply an amplitude modulation.

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))
elif 1 == signalType:
    '''
    START - atili additions
    '''

    path = 'D:/atili/MMIExt/Audacity/METU Recordings/hh2_breath/hh2_09_01.20.741_134_en.wav'
    # path = 'D:/atili/MMIExt/Audacity/METU Recordings/hh2_breath/hh2_09_01.20.741_134_en3_16bit.wav'
    # path = 'D:/atili/MMIExt/Audacity/Initial Breath Examples/bb_tr001_cigdem_07.wav'

    fs, signal = scipy.io.wavfile.read(path)
    print(fs, signal, signal.dtype)
    t = np.arange(len(signal)) / fs
    '''
    END - atili additions
    '''
else:
    print('TYPE NOT DEFINED')

# The amplitude envelope is given by magnitude of the analytic signal. The instantaneous frequency can be obtained by
#  differentiating the instantaneous phase in respect to time. The instantaneous phase corresponds to the phase angle
#  of the analytic signal.

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
# ax1.set_ylim(0.0, 120.0)
plt.show()


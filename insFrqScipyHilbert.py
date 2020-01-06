# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp, stft
# atili: addition imports
import utils

# In this example we use the Hilbert transform to determine the amplitude envelope and instantaneous frequency
# of an amplitude-modulated signal.

# Note that End-point limitation exists in the example

if 0 == utils.signalType:
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    # We create a chirp of which the frequency increases from 20 Hz to 100 Hz and apply an amplitude modulation.

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))
elif 1 == utils.signalType:
    '''
    START - atili additions
    '''

    path = '/METU Recordings/hh2_breath/hh2_09_01.20.741_134_en.wav'
    # path = '/METU Recordings/hh2_breath/hh2_09_01.20.741_134_en3_16bit.wav'
    # path = '/METU Recordings/Initial Breath Examples/bb_tr001_cigdem_07.wav'

    path = utils.prefix + path
    fs, signal = utils.readMonoWav(path)

    print(fs, 'Hz, Size=', signal.dtype, '*', len(signal), 'bytes, Array:', signal)
    t = np.arange(len(signal)) / fs
    '''
    END - atili additions
    '''
else:
    print('TYPE NOT DEFINED')
    signal = None
    fs = None
    t = None


if signal is not None:
    # The amplitude envelope is given by magnitude of the analytic signal. The instantaneous frequency can be obtained
    # by differentiating the instantaneous phase in respect to time. The instantaneous phase corresponds to the phase
    # angle of the analytic signal.

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi) * fs)

    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax0.plot(t, signal, label='signal')
    # ax0.plot(t, amplitude_envelope, label='envelope')
    # ax0.legend()
    ax0.set_ylabel('Amplitude')
    ax0.set_title('Sinusoidal Chirp Signal with increasing frequency')
    ax1 = fig.add_subplot(312)
    ax1.plot(t[1:], instantaneous_frequency)
    ax1.set_title('Instantaneous Frequency')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_ylim(0.0, 120.0)
    ax2 = fig.add_subplot(313)
    f, t, Zxx = stft(signal, fs, nperseg=32)
    ax2.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=1)
    ax2.set_title('STFT Magnitude')
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_ylim(0.0, 120.0)
    plt.tight_layout()
    plt.savefig('./plots/hht_stft.svg')
    plt.savefig('./plots/hht_stft.png')
    plt.show()


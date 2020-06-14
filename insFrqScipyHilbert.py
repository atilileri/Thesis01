import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp, stft
from datetime import datetime
import scipy.io.wavfile

scriptStartDateTime = datetime.now().strftime('%Y%m%d_%H%M%S')
useChirp = False

if useChirp:
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    # We create a chirp of which the frequency increases from 20 Hz to 100 Hz and apply an amplitude modulation.

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))
else:
    # !!!!! IMPORTANT NOTE !!!!! : sample file must have dtype of "Integer" for stft() to work
    path = 'D:/sampleBreathInt16.wav'

    fs, signal = scipy.io.wavfile.read(path)

    print(fs, 'Hz, Size=', signal.dtype, '*', len(signal), 'bytes, Array:', signal)
    t = np.arange(len(signal)) / fs

if signal is not None:

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi) * fs)
    # plot - compared
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax0.plot(t, signal, label='signal')
    ax0.set_ylabel('Amplitude')
    if useChirp:
        ax0.set_title('Sinusoidal Chirp Signal with increasing frequency')
    else:
        ax0.set_title('Breath Sample')
    ax1 = fig.add_subplot(312)
    ax1.plot(t[1:], instantaneous_frequency)
    ax1.set_title('Instantaneous Frequency')
    ax1.set_ylabel('Frequency [Hz]')
    if useChirp:
        ax1.set_ylim(0.0, 120.0)
    else:
        ax1.set_ylim(0.0, fs/2.0)
    ax2 = fig.add_subplot(313)
    f, t, Zxx = stft(signal, fs, nperseg=512)  # note - ai : modify according to signal for best view
    ax2.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=1)
    ax2.set_title('STFT Magnitude')
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')
    if useChirp:
        ax2.set_ylim(0.0, 120.0)
    else:
        ax2.set_ylim(0.0, fs/2.0)
    plt.tight_layout()
    plt.savefig('./plots/hht_stft_'+scriptStartDateTime+'.svg')
    plt.savefig('./plots/hht_stft_'+scriptStartDateTime+'.png')
    plt.show()


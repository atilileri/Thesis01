import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import gc
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D


def compareSignals(folder, fileName1, fileName2):
    print('Comparing', fileName1, 'and', fileName2, 'at', folder, flush=True)
    gc.collect()
    fs, in1 = scipy.io.wavfile.read(folder + '/' + fileName1)
    fs, in2 = scipy.io.wavfile.read(folder + '/' + fileName2)
    in1 = np.swapaxes(in1, 0, 1)
    in2 = np.swapaxes(in2, 0, 1)

    print('Plotting...', flush=True)
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')

    for chIdx in range(len(in1)):
        ax = axes[chIdx // 2, chIdx % 2]
        # ax.plot(20 * np.log10(abs(np.fft.rfft(in1[chIdx]))))
        # ax.plot(20 * np.log10(abs(np.fft.rfft(in2[chIdx]))))
        ax.magnitude_spectrum(in1[chIdx], Fs=48000, scale='dB')
        ax.magnitude_spectrum(in2[chIdx], Fs=48000, scale='dB')

        ax.set_title('Channel #%d' % (chIdx+1))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([0, -50, -100, -150, -200, -250])
        ax.grid(linestyle=':')

        gc.collect()

    fig.text(0.5, 0.01, 'Magnitude(dB)', ha='center')
    fig.text(0.005, 0.5, 'Frequency(Hz)', va='center', rotation='vertical')
    # create legend
    plt.figlegend(handles=[
        Line2D([0], [0], color='#1f77b4', label='Original'),
        Line2D([0], [0], color='#ff7f0e', label='Filtered')
    ], loc='center')
    plt.show()


# folderpath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/Recordings_Max/aa'
folderpath = 'D:'

fileName = 'sampleBreathF.wav'

compareSignals(folderpath, '_original.'.join(fileName.split('.')), '_filtered.'.join(fileName.split('.')))

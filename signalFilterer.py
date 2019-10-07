import scipy.io.wavfile
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import gc


def filterRec(folder, fileName, save, plot):
    print('Filtering', fileName, 'at', folder, flush=True)
    gc.collect()
    fs, inputAllChannels = scipy.io.wavfile.read(folder + '/' + fileName)
    inputAllChannels = np.swapaxes(inputAllChannels, 0, 1)

    if plot:  # todo - ai : ask hh to calculate plots
        ch0 = inputAllChannels[0]
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        ax1.plot(range(len(ch0)), ch0)
        ax2.magnitude_spectrum(ch0, Fs=fs, scale='dB')

        A = np.fft.rfft(ch0)
        M = 20 * np.log10(abs(A))

        x = np.fft.rfftfreq(len(ch0), d=1/48000)
        plt.xscale('log')
        ax3.plot(x, M, linewidth=0.4)
        # bot = [-100]*len(M)
        ax3.fill(x, M)

        plt.show()

    outputAllChannels = list()
    numTaps = 1025
    h = scipy.signal.firwin(numtaps=numTaps, cutoff=70, pass_zero=False, fs=48000)
    for chIdx in range(len(inputAllChannels)):
        outputAllChannels.append(scipy.signal.lfilter(h, [1], inputAllChannels[chIdx])[numTaps//2:])
        # todo - hh : are removing amounts are true? (numTaps//2)

    # restore channel index order
    outputAllChannels = np.swapaxes(outputAllChannels, 0, 1)
    inputAllChannels = np.swapaxes(inputAllChannels, 0, 1)[numTaps//2:]  # compensation for filter delay
    gc.collect()
    if save:
        fileSaveName = '_filtered.'.join(fileName.split('.'))
        print(fileSaveName, 'saved to', folder)
        scipy.io.wavfile.write(folder + '/' + fileSaveName, fs, outputAllChannels)
        print(np.shape(outputAllChannels))
        del outputAllChannels
        fileSaveName = '_original.'.join(fileName.split('.'))
        print(fileSaveName, 'saved to', folder)
        scipy.io.wavfile.write(folder + '/' + fileSaveName, fs, inputAllChannels)
        print(np.shape(inputAllChannels))
        del inputAllChannels
    gc.collect()


# folderpath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/Recordings_Max/'
folderpath = 'D:/'

for root, directories, files in os.walk(folderpath):
    gc.collect()
    for file in files:
        gc.collect()
        if ('M.wav' in file) \
                and ('_original.wav' not in file) \
                and ('_filtered.wav' not in file):  # todo - ai : refactor here. dont forget 'M'
            filterRec(root, file, save=True, plot=False)
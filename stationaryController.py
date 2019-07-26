import os
import scipy.io.wavfile
import scipy.signal as sp
import pyhht.emd
import numpy as np
import utils
import gc
import sys

folderBreaths = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_max_sample_set/'
numberOfImfs = 9

print('Extracting Features at: %s' % folderBreaths)

for rootPath, directories, files in os.walk(folderBreaths):
    for filename in sorted(files):
        if '.wav' in filename:
            print()
            print('Examining Features of %s ...' % filename)
            filepath = rootPath + filename
            samplingRate, audioData = scipy.io.wavfile.read(filepath)

            audioData = np.swapaxes(audioData, 0, 1)

            for chIdx in range(len(audioData)):
                print('Channel %d Signal: ' % (chIdx+1), end='')
                utils.isStationary(audioData[chIdx])

                utils.blockPrint()  # block unwanted prints from emd implementation
                # detail params: threshold_1=0.000001, threshold_2=0.00001
                decomposer = pyhht.emd.EMD(audioData[chIdx], n_imfs=numberOfImfs)
                decomposedSignals = decomposer.decompose()
                utils.enablePrint()
                imfs = decomposedSignals[:-1]  # last element is residue
                # print(np.shape(imfs))

                # Calculate Magnitudes from IMFs(before normalization)
                for i in range(len(imfs)):
                    print('==IMF %d:\t ' % (i+1), end='')
                    utils.isStationary(imfs[i])
                    hx = sp.hilbert(imfs[i])
                    # magnitude
                    mag = np.abs(hx)  # magnitudes obtained before normalization
                    # mags.append(mag)
                    # phase
                    phx = np.unwrap(np.arctan2(hx.imag, hx.real))
                    # phases.append(phx)
                    # instant frequency
                    diff = np.diff(phx)
                    tempInstf = (samplingRate / (2 * np.pi)) * diff
                    print('==Instf %d: ' % (i+1), end='')
                    utils.isStationary(tempInstf)
                    # instfs.append(tempInstf)

                # instfs = np.array(instfs).swapaxes(0, 1)
                # mags = np.array(mags).swapaxes(0, 1)[:-1]
                # phases = np.array(phases).swapaxes(0, 1)[:-1]

##
# Feature Extractor
##

import os
import scipy.io.wavfile
import scipy.signal as sp
import pyhht.emd
import numpy as np
import utils
import gc
import sys

folders = [
    # 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10_1/',
    'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10_2_cont/'
           ]
# folderBreaths = 'E:/atili/Datasets/BreathDataset/Processed_Small/breaths_00000000_000000/'
# folderBreaths = 'E:/atili/Datasets/BreathDataset/Processed_Small/breaths_20190608_143805/'

sampRateMode = 48

if 48 == sampRateMode:
    numberOfImfs = 9
    srDivider = 1
elif 8 == sampRateMode:
    numberOfImfs = 7
    srDivider = 6
else:
    print('Invalid Sampling Rate Mode:', sampRateMode)
    sys.exit()

for folderBreaths in folders:
    print('Extracting Features at: %s' % folderBreaths)

    for rootPath, directories, files in os.walk(folderBreaths):
        for filename in sorted(files):
            if '.wav' in filename:
                print()
                print('Extracting Features of %s ...' % filename)
                filepath = rootPath + filename
                samplingRate, audioData = scipy.io.wavfile.read(filepath)
                if srDivider > 1:
                    samplingRate /= srDivider
                    audioData = audioData[::srDivider]
                gc.collect()
                audioData = np.swapaxes(audioData, 0, 1)

                spectos = []
                imfFeatures = []
                for eachChannel in audioData:
                    utils.blockPrint()  # block unwanted prints from emd implementation
                    # detail params: threshold_1=0.000001, threshold_2=0.00001
                    decomposer = pyhht.emd.EMD(eachChannel, n_imfs=numberOfImfs)
                    decomposedSignals = decomposer.decompose()
                    utils.enablePrint()
                    imfs = decomposedSignals[:-1]  # last element is residue
                    # print(np.shape(imfs))

                    # Calculate Magnitudes from IMFs(before normalization)
                    mags = []
                    instfs = []
                    phases = []
                    for imf in imfs:
                        hx = sp.hilbert(imf)
                        # magnitude
                        mag = np.abs(hx)  # magnitudes obtained before normalization
                        mags.append(mag)
                        # phase
                        phx = np.unwrap(np.arctan2(hx.imag, hx.real))
                        phases.append(phx)
                        # instant frequency
                        diff = np.diff(phx)
                        tempInstf = (samplingRate / (2 * np.pi)) * diff
                        instfs.append(tempInstf)

                    nMags = []
                    nInstfs = []
                    nPhases = []
                    # Normalization according to:
                    # http://www.ancad.com.tw/newsletter/test
                    # /On%20instantaneous%20frequency%20calculation%20o
                    # /On%20instantaneous%20frequency%20calculation%20o.htm
                    for imfEach in imfs:
                        # get a copy for normalization
                        normalized = imfEach.copy()
                        envelope = normalized  # pseudo envelope for while-loop entrance
                        while len(set(envelope)) > 1:  # loop until envelope is all the same
                            # 1. Take absolute value of IMF.
                            absImf = np.abs(normalized)
                            # 2. Find extrema.
                            peaks, _ = sp.find_peaks(absImf, height=0)  # peaks over 0
                            # 2.fix. Add first and last indexes for envelope calculations
                            peaks = np.concatenate(([0], peaks, [len(absImf) - 1]))
                            # 3. Based on these extrema, construct envelope.
                            envelope = np.interp(range(len(absImf)), peaks, absImf[peaks])
                            # 4. Normalize IMF using the envelope. The FM part of signal becomes almost equal amplitude.
                            normalized = np.divide(normalized, envelope)
                            # 5. Repeat process 2-4 after the amplitude of normalized IMF retains
                            # a straight line with identical value.

                        # Calculate Instant Frequencies from normalized IMFs
                        hx = sp.hilbert(normalized)
                        # normalized magnitude
                        mag = np.abs(hx)  # magnitudes obtained before normalization
                        nMags.append(mag)
                        # normalized phase
                        phx = np.unwrap(np.arctan2(hx.imag, hx.real))
                        nPhases.append(phx)
                        # normalized instant frequency
                        diff = np.diff(phx)
                        tempInstf = (samplingRate / (2 * np.pi)) * diff
                        nInstfs.append(tempInstf)

                    inputChannel = []
                    instfs = np.array(instfs).swapaxes(0, 1)
                    mags = np.array(mags).swapaxes(0, 1)[:-1]
                    phases = np.array(phases).swapaxes(0, 1)[:-1]
                    nInstfs = np.array(nInstfs).swapaxes(0, 1)
                    nMags = np.array(nMags).swapaxes(0, 1)[:-1]
                    nPhases = np.array(nPhases).swapaxes(0, 1)[:-1]
                    # print(instfs.shape, mags.shape, phases.shape)
                    # print(nInstfs.shape, nMags.shape, nPhases.shape)
                    for i in range(len(instfs)):
                        inputChannel.append([instfs[i], mags[i], phases[i], nInstfs[i], nMags[i], nPhases[i]])
                    imfFeatures.append(inputChannel)

                    # spectogram
                    f, scriptStartDateTime, z = sp.stft(eachChannel, fs=48000)
                    specto = np.abs(z)
                    specto = np.array(specto).swapaxes(0, 1)
                    # print(specto.shape)
                    spectos.append(specto)

                # shape is (channel, sampleCount, 6(instf, mag, phase and normalized versions), imfCount)
                # Swap axes to (sampleCount, channel, 6, imfCount)
                imfFeatures = np.array(imfFeatures).swapaxes(0, 1)
                spectos = np.array(spectos).swapaxes(0, 1)
                # Prepare input file for LSTM
                folderpath = os.path.dirname(filepath)
                foldername = os.path.basename(folderpath).replace('breaths', 'inputsFrom')
                dataSaveFolder = os.path.dirname(folderpath) + '/' + foldername

                imfFileExt = '.imf'
                if 48 == sampRateMode:
                    imfFileExt = imfFileExt + '48'
                elif 8 == sampRateMode:
                    imfFileExt = imfFileExt + '08'
                else:
                    print('Invalid Sampling Rate Mode:', sampRateMode)
                    sys.exit()
                savefilename = dataSaveFolder + '/' + filename.replace('.wav', imfFileExt)
                # create folder if not existed
                if not os.path.exists(os.path.dirname(savefilename)):
                    os.makedirs(os.path.dirname(savefilename))

                inputShape = np.shape(imfFeatures)
                # print(inputShape)
                if inputShape[1:] == (4, 6, numberOfImfs):
                    # save imf feature input file
                    utils.saveData(imfFeatures, savefilename)
                    print('Prepared: ' + savefilename, inputShape)
                    if 48 == sampRateMode:
                        # save spectogram input file
                        inputShape = np.shape(spectos)
                        # print(inputShape)
                        savefilename = dataSaveFolder + '/' + filename.replace('.wav', '.spct48')
                        utils.saveData(spectos, savefilename)
                        print('Prepared: ' + savefilename, inputShape)
                    elif 8 == sampRateMode:
                        pass
                    else:
                        print('Invalid Sampling Rate Mode:', sampRateMode)
                        sys.exit()
                else:
                    print('')
                    print('--------------Broken: ' + savefilename, inputShape)
                    print('')

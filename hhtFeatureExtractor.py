##
# Feature Extractor
##

import os
import scipy.io.wavfile
import scipy.signal as sp
import pyhht.emd
import numpy as np
import utils

# folderBreaths = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_mini_sample_set/'
folderBreaths = 'E:/atil/Processed/breaths_20190504_181608/'
numberOfImfs = 9

for rootPath, directories, files in os.walk(folderBreaths):
    for filename in files:
        if '.wav' in filename:
            print('Extracting Features of:', filename, '\t\t @', rootPath)
            filepath = rootPath + filename
            samplingRate, audioData = scipy.io.wavfile.read(filepath)
            audioData = np.swapaxes(audioData, 0, 1)

            # inputFile = []
            inputFile2 = []
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
                for imf in imfs:
                    hx = sp.hilbert(imf)
                    mag = np.abs(hx)  # magnitudes are obtained before normalization
                    mags.append(mag)

                instfs = []
                # Normalization according to:
                # http://www.ancad.com.tw/newsletter/test
                # /On%20instantaneous%20frequency%20calculation%20o/On%20instantaneous%20frequency%20calculation%20o.htm
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
                    phx = np.unwrap(np.arctan2(hx.imag, hx.real))
                    diff = np.diff(phx)
                    tempInstf = (samplingRate / (2 * np.pi)) * diff
                    instfs.append(tempInstf)

                # inputChannel = []
                inputChannel2 = []
                swappedInstfArr = np.array(instfs).swapaxes(0, 1)
                swappedMagArr = np.array(mags).swapaxes(0, 1)[:-1]
                # print(swappedInstfArr.shape, swappedMagArr.shape)
                for i in range(len(swappedInstfArr)):
                    # inputChannel.append({'instantFrequencies': swappedInstfArr[i], 'magnitudes': swappedMagArr[i]})
                    inputChannel2.append([swappedInstfArr[i], swappedMagArr[i]])
                # inputFile.append(inputChannel)
                inputFile2.append(inputChannel2)

            # shape is (channel, sampleCount, 2(instf and mag), imfCount)
            # Swap axes to (sampleCount, channel, 2, imfCount)
            # inputFile = np.array(inputFile).swapaxes(0, 1)
            inputFile2 = np.array(inputFile2).swapaxes(0, 1)
            # Prepare input file for LSTM
            folderpath = os.path.dirname(filepath)
            foldername = os.path.basename(folderpath).replace('breaths', 'inputsFrom')
            dataSaveFolder = os.path.dirname(folderpath) + '/' + foldername
            savefilename = dataSaveFolder + '/' + filename.replace('.wav', '.inp')
            if not os.path.exists(os.path.dirname(savefilename)):
                os.makedirs(os.path.dirname(savefilename))
            # print(np.shape(inputFile))
            # utils.saveData(inputFile, savefilename)
            utils.saveData(inputFile2, savefilename + '2')
            print('Prepared: ' + savefilename + '2')

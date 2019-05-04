"""
Atil Ilerialkan
Instant Frequency Spectogram Based on HHT
"""
import pyhht.emd
import PyEMD
# import pyeemd
import utils
import matplotlib.pyplot as plt
import scipy.signal as sp
import os
import numpy as np
from enum import Enum
import pyperclip as cl
np.set_printoptions(edgeitems=50)

if __name__ == '__main__':
    saveFolder = ''
    # filepath = './METU Recordings/hh2_48kHz_Mono_32bitFloat.wav'
    # filepath = './METU Recordings/hh2_breath/hh2_04_00.34.164_270_en.wav'
    # filepath = './METU Recordings/hh2_breath/hh2_09_01.20.741_134_en2.wav'
    # filepath = './METU Recordings/hh2_breath/hh2_09_01.20.741_134_en3_16bit.wav'
    # filepath = './METU Recordings/hh2_breath/hh2_23_03.02.050_149_tr.wav'
    # filepath = './METU Recordings/hh2_withEdges/hh2_random001_noised.wav'

    # saveFolder += 'density/bg/'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_00.55.000-571_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.12.437-204_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.22.210-876_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.31.170-404_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.37.775-506_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.45.197-192_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.59.505-447_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.11.400-373_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.32.472-282_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.36.313-440_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.37.779-554_bg.wav'
    # filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.39.465-329_bg.wav'

    saveFolder += 'density/breath/'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.32.850-525_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.40.871-424_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.43.460-591_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.52.458-437_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.59.523-922_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.05.275-933_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.13.717-623_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.24.215-555_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.27.825-780_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.50.001-491_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.57.027-852_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_02.26.115-493_breath.wav'
    # filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_02.39.838-462_breath.wav'
    # filepath = './METU Recordings/Dataset/BurcakYildirim/by01_01.12.305-408.wav'
    # filepath = './METU Recordings/Dataset/DemirSiginan/ds01_01.12.708-419.wav'
    filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_20190503_033537/ba01_1_1.212-0.201.wav'

    # saveFolder += 'density/voiced/'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_00.21.576-326_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_00.24.897-274_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_00.31.490-324_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_00.42.814-516_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_01.18.383-252_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_01.32.227-273_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_01.35.925-301_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_02.59.372-200_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_03.18.690-234_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_03.35.168-364_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_03.47.911-201_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_04.25.179-287_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_04.35.304-254_v.wav'
    # filepath = './TED Recordings/BillGates_2009/voiced/BillGates_2009_05.04.223-208_v.wav'

    # Parameter options
    useCenterFrame = False  # Whether to use a centered portion of the signal (requires centerFrameLenBySamples)
    useImfNormalization = True  # Whether to use normalized data on Instant Freq Spectogram
    plotSpec = False
    plotVect = False
    # Start -  Only useful if plotVect is true
    colorAsMagnitude = False  # color is used as magnitude or imfNo(alpha can be magnitude)
    alphaAsMagnitude = False  # Whether to use alpha of rgba as magnitude. Useful if colorAsMagnitude is False
    # End - Only useful if plotVect is true
    plotInstf = False
    plotImfs = False
    plotDensity = False
    prepareLSTMInputFile = True


    class EMDSelection(Enum):  # Different EMD algorithms
        pyHht = 1
        pyEmdVanilla = 2
        pyEmdEnsemble = 3   # Takes too long and never converges to zero IMF extraction(at least 1 IMF + 1 residue)
        pyEmdComplete = 4   # Not working
        pyEemd = 5          # Not working

    emdSel = EMDSelection.pyHht

    # Read File
    sampRate, sig = utils.readMonoWav(filepath)
    filename = str(os.path.basename(filepath).rsplit('.', 1)[0])
    label = filename.split('0')[0]

    if useCenterFrame:  # Use only a center window with a spesified length(centerFrameLenBySamples)
        centerFrameLenBySamples = min(int(sampRate * 0.2), len(sig))  # center frame is 200 ms
        mid = len(sig)//2
        sig = sig[mid-(centerFrameLenBySamples//2):mid+(centerFrameLenBySamples//2)]

    print('----- File Info ------')
    print('FileName:', filename)
    print('SamplingRate:', sampRate, 'samples per second')
    print('Length:', len(sig), 'samples(', len(sig)/sampRate, 'seconds )')
    print('----------------------')

    imfs = []
    residue = sig.copy()
    # Calculate IMFs (in a loop) for the signal
    while True:
        decomposedSignals = []
        if emdSel is EMDSelection.pyHht:
            decomposer = pyhht.emd.EMD(residue)  # detail params: threshold_1=0.000001, threshold_2=0.00001
            decomposedSignals = decomposer.decompose()
        elif emdSel is EMDSelection.pyEmdVanilla:
            decomposer = PyEMD.EMD()
            decomposedSignals = decomposer(residue)
        elif emdSel is EMDSelection.pyEmdEnsemble:
            decomposer = PyEMD.EEMD()
            decomposedSignals = decomposer(residue)
        elif emdSel is EMDSelection.pyEmdComplete:
            decomposer = PyEMD.CEEMDAN()
            decomposedSignals = decomposer(residue)
        elif emdSel is EMDSelection.pyEemd:
            # imfs = pyeemd.ceemdan(residue, S_number=4, num_siftings=50)
            break

        if len(decomposedSignals) > 1:
            imfs.extend(decomposedSignals[:-1])
            residue = decomposedSignals[-1]  # last element is residue
            if len(decomposedSignals) == 2:  # if only 1 imf is extracted, try no more
                break
        else:
            break

    print(len(imfs), 'IMFs extracted.')
    print('----------------------')

    # Calculate Instant Frequencies from IMFs(before normalization)
    instfs = []
    mags = []
    for imf in imfs:
        hx = sp.hilbert(imf)
        mag = np.abs(hx)  # magnitudes are obtained before normalization
        phx = np.unwrap(np.arctan2(hx.imag, hx.real))
        diff = np.diff(phx)
        instf = (sampRate / (2 * np.pi)) * diff

        instfs.append(instf)
        mags.append(mag)
    print('Instant Frequencies calculated on IMFs (before normalization).')

    # Normalization according to:
    # http://www.ancad.com.tw/newsletter/test
    # /On%20instantaneous%20frequency%20calculation%20o/On%20instantaneous%20frequency%20calculation%20o.htm
    imfsNorm = []
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
            peaks = np.concatenate(([0], peaks, [len(absImf)-1]))
            # 3. Based on these extrema, construct envelope.
            envelope = np.interp(range(len(absImf)), peaks, absImf[peaks])
            # 4. Normalize IMF using the envelope. The FM part of signal becomes almost equal amplitude.
            normalized = np.divide(normalized, envelope)
            # 5. Repeat process 2-4 after the amplitude of normalized IMF retains a straight line with identical value.
        imfsNorm.append(normalized)
    print('IMFs normalized')

    # Calculate Instant Frequencies from normalized IMFs(after normalization)
    instfsNorm = []
    for eachImf in imfsNorm:
        hx2 = sp.hilbert(eachImf)
        phx2 = np.unwrap(np.arctan2(hx2.imag, hx2.real))
        diff2 = np.diff(phx2)
        tempInstf2 = (sampRate / (2 * np.pi)) * diff2
        instfsNorm.append(tempInstf2)
    print('Instant Frequencies calculated on normalized IMFs.')

    # Prepare spectogram parameters
    freqBinDivider = 10
    freqBinCount = int(np.ceil(sampRate / (freqBinDivider * 2)))
    specs = np.zeros(shape=(len(sig), freqBinCount))

    if useImfNormalization:
        instfArr = instfsNorm
    else:
        instfArr = instfs
    # Prepare spectogram data
    for frameIdx in range(len(sig) - 1):
        for freqIdx in range(len(instfArr)):
            freq = instfArr[freqIdx][frameIdx]
            specs[frameIdx][int(np.floor(abs(freq) / freqBinDivider))] += mags[freqIdx][frameIdx]
    print('Spectogram prepared.')

    # Specialize density analysis area
    densityAnalysisArea = specs[:, freqBinCount//4:-freqBinCount//4]
    weightedDensityAnalysisArea = np.multiply(densityAnalysisArea.astype(bool).astype(int),
                                              np.linspace(1, 10, np.shape(densityAnalysisArea)[1]))

    # Calculate densities
    density = np.count_nonzero(densityAnalysisArea, axis=1)
    avgDensity = np.average(density)

    weiDensity = np.sum(weightedDensityAnalysisArea, axis=1)
    avgWeiDensity = np.average(weiDensity)
    print('Density Analysis is done')

    if prepareLSTMInputFile:
        # Prepare input file for LSTM
        print('Preparing Input file for:', label)
        folderpath = os.path.dirname(filepath)
        foldername = os.path.basename(folderpath).split('_')
        foldername[0] = 'inputs'
        dataSaveFolder = os.path.dirname(folderpath)+'/'+'_'.join(foldername)
        inputFile = []
        inputFile2 = []
        swappedInstfArr = np.array(instfArr).swapaxes(0, 1)
        swappedMagArr = np.array(mags).swapaxes(0, 1)[:-1]
        print(swappedInstfArr.shape, swappedMagArr.shape)
        for i in range(len(swappedInstfArr)):
            inputFile.append({'instantFrequencies': swappedInstfArr[i], 'magnitudes': swappedMagArr[i]})
            inputFile2.append([swappedInstfArr[i], swappedMagArr[i]])
        savefilename = dataSaveFolder+'/'+filename+'.inp'
        if not os.path.exists(os.path.dirname(savefilename)):
            os.makedirs(os.path.dirname(savefilename))
        print(np.shape(inputFile))
        utils.saveData(inputFile, savefilename)
        utils.saveData(np.array(inputFile2), savefilename+'2')
        print(filename+'.inp prepared.')
    print('----------------------')

    print('plotting...')

    if plotSpec:
        specsAspectRatio = specs.shape[0] / specs.shape[1]  # for better print size and look
        # Make plot with vertical (default) colorbar
        fig, ax = plt.subplots(figsize=(15 * specsAspectRatio, 15))

        data = np.swapaxes(specs, 0, 1)
        data = np.ma.masked_values(data, 0)

        cmap = plt.cm.magma
        cmap.set_bad(color='white')

        cax = ax.imshow(data, interpolation='nearest', cmap=cmap, origin='lower')
        ax.set_title(filename + ' Magnitudes of IMFs')

        # Add colorbar
        cbar = fig.colorbar(cax)
        plt.savefig('plots/graphMag_' + filename + '.png')
        plt.show()

    if plotVect:
        plt.figure(figsize=(7, 3))
        if colorAsMagnitude:
            myCmap = plt.cm.magma
            myCmap.set_bad(color='white')
        else:  # color as imfNo
            # Get the colormap colors
            cmap = plt.cm.Dark2
            cmap2 = plt.cm.Set1
            myCmap = np.concatenate((cmap(np.arange(cmap.N)), cmap2(np.arange(cmap2.N))))
        for i in range(len(instfArr)):
            # plt.plot(i)
            xValues = np.divide(range(len(instfArr[i])), sampRate)
            yValues = abs(instfArr[i])
            if colorAsMagnitude:
                colors = mags[i][:]
                plt.scatter(xValues, yValues, marker=",", s=0.001, c=colors,
                            linewidths=0.001, label="Instf #"+str(i), zorder=i, cmap=myCmap)
                plt.colorbar(ticks=[])
            else:  # color as imfNo
                colors = np.array([myCmap[i]] * len(instfArr[i]))
                if alphaAsMagnitude:  # alpha is magnitude
                    alphaValues = mags[i] / np.max(mags[i])
                    colors[:, 3] = alphaValues[:len(instfArr[i])]
                plt.scatter(xValues, yValues, marker=",", s=0.001,
                            linewidths=0.001, label="Instf #"+str(i), zorder=i, color=colors)
        plt.savefig('plots/graphMag_' + filename + '.svg')
        plt.savefig('plots/graphMag_' + filename + '_fromVector.png', dpi=2000)
        plt.savefig('plots/graphMag_' + filename + '.pdf', format='pdf')
        plt.show()

    if plotInstf:
        # Instant Frequencies for all IMFs combined
        for i in range(len(instfs)):
            plt.title(filename + ' Frequencies of Instfs')
            plt.plot(np.divide(range(len(instfs[i])), sampRate), instfs[i], linewidth=0.5)
            plt.savefig("plots/graphAllInstf_" + filename + ".svg")
            plt.savefig("plots/graphAllInstf_" + filename + ".png")
        plt.show()

        # Instant Frequencies for each IMF
        for i in range(len(instfs)):
            plt.title(filename + ' Frequencies of IMF#' + str(i+1))
            plt.plot(np.divide(range(len(instfs[i])), sampRate), instfs[i], linewidth=0.5)
            plt.savefig("plots/graphInstf_" + str(i+1) + "_" + filename + ".svg")
            plt.savefig("plots/graphInstf_" + str(i+1) + "_" + filename + ".png")
            plt.show()

        # Normalized Instant Frequencies for all IMFs combined
        for i in range(len(instfsNorm)):
            plt.title(filename + ' Frequencies of Normalized Instfs')
            plt.plot(np.divide(range(len(instfs[i])), sampRate), instfs[i], linewidth=0.5)
            plt.savefig("plots/graphAllInstfNorm_" + filename + ".svg")
            plt.savefig("plots/graphAllInstfNorm_" + filename + ".png")
        plt.show()

        # Normalized Instant Frequencies for each IMF
        for i in range(len(instfsNorm)):
            plt.title(filename + ' Frequencies of Normalized IMF#' + str(i+1))
            plt.plot(np.divide(range(len(instfsNorm[i])), sampRate), instfsNorm[i], linewidth=0.5)
            plt.savefig("plots/graphInstfNorm_" + str(i+1) + "_" + filename + ".svg")
            plt.savefig("plots/graphInstfNorm_" + str(i+1) + "_" + filename + ".png")
            plt.show()

        # Normalized and Absoluted Instant Frequencies for all IMFs combined
        for i in range(len(instfsNorm)):
            plt.title(filename + ' Frequencies of Norm + Abs Instfs')
            plt.plot(np.divide(range(len(instfs[i])), sampRate), abs(instfs[i]), linewidth=0.5)
            plt.savefig("plots/graphAllInstfNormAbs_" + filename + ".svg")
            plt.savefig("plots/graphAllInstfNormAbs_" + filename + ".png")
        plt.show()

        # Normalized and Absoluted Instant Frequencies for each IMF
        for i in range(len(instfsNorm)):
            plt.title(filename + ' Frequencies of Norm + Abs IMF#' + str(i+1))
            plt.plot(np.divide(range(len(instfsNorm[i])), sampRate), abs(instfsNorm[i]), linewidth=0.5)
            plt.savefig("plots/graphInstfNormAbs_" + str(i+1) + "_" + filename + ".svg")
            plt.savefig("plots/graphInstfNormAbs_" + str(i+1) + "_" + filename + ".png")
            plt.show()

    if plotImfs:
        for i in range(np.shape(imfs)[0]):
            # Plot Original Signal
            # plt.plot(np.divide(range(len(sig)), sampRate), sig,
            #          label="Orig", color='gray', linewidth=0.5, zorder=-3)
            # Plot Instant Frequencies
            plt.subplot(211)
            plt.title('IMF #' + str(i+1))
            plt.plot(np.divide(range(len(instfs[i])), sampRate), instfs[i],
                     label="Instf", color='green', linewidth=0.7, zorder=2)
            plt.plot(np.divide(range(len(instfsNorm[i])), sampRate), instfsNorm[i],
                     label="Instf from NormImfs", color='brown', linewidth=0.7, zorder=5)
            plt.grid(True, linestyle='dashed', color='darkred', linewidth=0.2, zorder=0)
            plt.xticks(np.arange(0, len(sig) / sampRate, step=0.1), rotation=90)
            # plt.ylabel('I')
            plt.legend()
            # Plot Imfs
            plt.subplot(212)
            # plt.title('IMF #' + str(i+1))
            plt.plot(np.divide(range(len(imfs[i])), sampRate), np.divide(imfs[i], max(imfs[i])),
                     label="Imf(normalized for plotting)", color='orange', linewidth=0.75, zorder=2)
            plt.plot(np.divide(range(len(imfsNorm[i])), sampRate), imfsNorm[i],
                     label="ImfNorm", color='blue', linewidth=0.75, zorder=1)
            # Plot Envelopes of Imfs
            # plt.plot(np.divide(range(len(imfsEnv[i])), sampRate), imfsEnv[i],
            #          label="Env")
            plt.grid(True, linestyle='dashed', color='darkred', linewidth=0.2, zorder=0)
            plt.xticks(np.arange(0, len(sig) / sampRate, step=0.1), rotation=90)
            plt.xlabel('Time (Seconds)')
            # plt.ylabel('Amplitude')
            plt.legend()
            plt.savefig("plots/graphImf"+str(i+1)+".svg")
            plt.savefig("plots/graphImf"+str(i+1)+".png")
            plt.show()

    if plotDensity:
        print('Avg Density:', avgDensity)
        print('Weighted Avg Density:', avgWeiDensity)
        cl.copy(str(avgDensity) + '\t' + str(avgWeiDensity))  # copy values to clipboard to paste excel :)

        plt.title(filepath)
        plt.plot(np.divide(range(len(density)), sampRate), density,
                 label="Average Density: " + str(avgDensity), linewidth=0.5)
        plt.legend()
        plt.savefig("plots/" + saveFolder + "density_" + filename + ".svg")
        plt.savefig("plots/" + saveFolder + "density_" + filename + ".png")
        plt.show()

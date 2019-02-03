"""
Code taken from:
https://pyhht.readthedocs.io/en/latest/apiref/pyhht.html#pyhht.utils.inst_freq
[2018.08.30]: Tried, works as example
"""
from pyhht.emd import EMD
from utils import readMonoWav
import matplotlib.pyplot as plt
import scipy.signal as sp
import os
import numpy as np
import pyperclip as cl
np.set_printoptions(edgeitems=50)

saveFolder = ''
# filepath = './METU Recordings/hh2_48kHz_Mono_32bitFloat.wav'
# filepath = './METU Recordings/hh2_breath/hh2_04_00.34.164_270_en.wav'
# filepath = './METU Recordings/hh2_breath/hh2_09_01.20.741_134_en2.wav'
# filepath = './METU Recordings/hh2_breath/hh2_09_01.20.741_134_en3_16bit.wav'
# filepath = './METU Recordings/hh2_breath/hh2_23_03.02.050_149_tr.wav'
# filepath = './METU Recordings/hh2_withEdges/hh2_random001_noised.wav'

saveFolder += 'density/bg/'
filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_00.55.000-571.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.12.437-204.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.22.210-876.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.31.170-404.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.37.775-506.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.45.197-192.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_01.59.505-447.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.11.400-373.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.32.472-282.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.36.313-440.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.37.779-554.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_02.39.465-329.wav'

# saveFolder += 'density/breath/'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.32.850-525.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.40.871-424.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.43.460-591.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.52.458-437.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.59.523-922.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.05.275-933.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.13.717-623.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.24.215-555.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.27.825-780.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.50.001-491.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_01.57.027-852.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_02.26.115-493.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_02.39.838-462.wav'

# saveFolder += 'density/non-voiced/'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_00.21.576-326.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_00.24.897-274.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_00.31.490-324.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_00.42.814-516.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_01.18.383-252.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_01.32.227-273.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_01.35.925-301.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_02.59.372-200.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_03.18.690-234.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_03.35.168-364.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_03.47.911-201.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_04.25.179-287.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_04.35.304-254.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_05.04.223-208.wav'

sampRate, sig = readMonoWav(filepath)
filename = str(os.path.basename(filepath).rsplit('.', 1)[0])

# todo - remove when mid signal tests completed
mid = len(sig)//2
centerFrameLenBySamples = min(int(sampRate * 0.2), len(sig))  # center frame is 200 ms
sig = sig[mid-(centerFrameLenBySamples//2):mid+(centerFrameLenBySamples//2)]

print('FileName:', filename)
print('Length:', len(sig), 'samples')
print('SamplingRate:', sampRate)
print('######')


readWholeFile = True  # set to False for trials only

if readWholeFile is False:
    # take a sample for better visualising, huge chunks crashes in hilbert()
    sig = sig[sampRate*0:sampRate*1]  # read only first 1 second

# sig = sig / max(sig)  # normalization

imfsAll = []
magAll = []
imfsAllNorm = []
imfsEnv = []
instfAll = []
partLen = min(sampRate // 1, len(sig))  # divide into 100 ms parts - shorter durations lead to less imfs
print('Each part is', partLen, 'samples and', partLen/sampRate, 'second(s)')
partCount = int(np.ceil(len(sig) / partLen))
print('Input signal is divided into', partCount, 'part(s)')

for i in range(partCount):
    startIndex = i*partLen
    endIndex = (i+1)*partLen
    # temporarily adding neighbor parts for more accurate calculations
    # todo - hh : only half or quarter of neighbor parts can be enough?
    if i > 0:  # if not first part
        startIndex -= partLen
    if i < partCount - 2:  # until second from last part
        endIndex += partLen
    if i == partCount - 2:  # second from last part (last part's len may not be partLen)
        endIndex += len(sig) % partLen
    part = sig[startIndex:endIndex]

    # calculate imfs for the part
    decomposer = EMD(part)
    imfsPart = decomposer.decompose()[:-1]  # last element is residue

    # calculate instant frequency for each imf of the part
    instfPart = []
    magPart = []
    truncatedImfs = []
    for imf in imfsPart:
        hx = sp.hilbert(imf)
        mag = np.abs(hx)
        phx = np.unwrap(np.arctan2(hx.imag, hx.real))
        tempInstf = sampRate / (2 * np.pi) * np.diff(phx)

        # removing neighbor parts after calculations
        if i > 0:  # not first part
            tempInstf = tempInstf[partLen:]
            mag = mag[partLen:]
            imf = imf[partLen:]
        if i < partCount - 2:  # until second from last part
            tempInstf = tempInstf[:-partLen]
            mag = mag[:-partLen]
            imf = imf[:-partLen]
        if i == partCount - 2:  # second from last part (last part's len may not be partLen)
            tempInstf = tempInstf[:-(len(sig) % partLen)]
            mag = mag[:-(len(sig) % partLen)]
            imf = imf[:-(len(sig) % partLen)]
        instfPart.append(tempInstf)
        magPart.append(mag)
        truncatedImfs.append(imf)

    # done with extra parts, set truncated imfs
    imfsPart = truncatedImfs

    # concatanate all parts' imfs together
    if not imfsAll:
        imfsAll = imfsPart
    else:
        print(np.shape(imfsAll), np.shape(imfsPart))
        while len(imfsAll) < len(imfsPart):  # if instf of this part has MORE rows
            imfsAll.append([0] * len(imfsAll[0]))  # add new imfs row to main list
        while len(imfsPart) < len(imfsAll):  # if instf of this part has LESS rows
            imfsPart.append([0] * len(imfsPart[0]))  # add new imf row to part
        imfsAll = list(np.concatenate((imfsAll, imfsPart), axis=1))

    # concatanate all parts' instant frequency together
    if not instfAll:
        instfAll = instfPart
    else:
        while len(instfAll) < len(instfPart):  # if instf of this part has MORE rows
            instfAll.append([0] * len(instfAll[0]))  # add new imfs row to main list
        while len(instfPart) < len(instfAll):  # if instf of this part has LESS rows
            instfPart.append([0] * len(instfPart[0]))  # add new imf row to part
        instfAll = list(np.concatenate((instfAll, instfPart), axis=1))

    # concatanate all parts' magnitudes together
    if not magAll:
        magAll = magPart
    else:
        while len(magAll) < len(magPart):  # if mag of this part has MORE rows
            magAll.append([0] * len(magAll[0]))  # add new imfs row to main list
        while len(magPart) < len(magAll):  # if mag of this part has LESS rows
            magPart.append([0] * len(magPart[0]))  # add new imf row to part
        magAll = list(np.concatenate((magAll, magPart), axis=1))

# Normalization according to:
# http://www.ancad.com.tw/newsletter/test
# /On%20instantaneous%20frequency%20calculation%20o/On%20instantaneous%20frequency%20calculation%20o.htm
for ii in range(len(imfsAll)):
    # 1. Take absolute value of IMF.
    absImf = np.abs(imfsAll[ii])
    # 2. Find extrema.
    peaks, _ = sp.find_peaks(absImf, height=0)  # peaks over 0
    peaks = np.concatenate(([0], peaks, [len(absImf)-1]))
    # 3. Based on these extrema, construct envelope.
    envelope = np.interp(range(len(absImf)), peaks, absImf[peaks])
    imfsEnv.append(envelope)
    # 4. Normalize IMF using the envelope. The FM part of signal becomes almost equal amplitude.
    imfsAllNorm.append(np.divide(imfsAll[ii], envelope))
    # 5. Repeat process 2-4 after the amplitude of normalized IMF retains a straight line with identical value.
    # Not needed for now

# for ii in range(len(imfsAll)):
#     imfsAll[ii] /= imfsEnv[ii]

instfAllNorm = []
for eachImf in imfsAllNorm:
    hx2 = sp.hilbert(eachImf)
    phx2 = np.unwrap(np.arctan2(hx2.imag, hx2.real))
    tempInstf2 = sampRate / (2 * np.pi) * np.diff(phx2)
    instfAllNorm.append(tempInstf2)

freqBinDivider = 10
freqBinCount = int(np.ceil(sampRate / (freqBinDivider * 2)))
specs = np.zeros(shape=(len(sig), freqBinCount))

for sampleIdx in range(len(sig)-1):
    for index2d in range(len(instfAllNorm)):
        freq = instfAllNorm[index2d][sampleIdx]
        specs[sampleIdx][int(np.floor(abs(freq) / freqBinDivider))] += magAll[index2d][sampleIdx]

densityAnalysisArea = specs[:, freqBinCount//4:-freqBinCount//4]
weightedDensityAnalysisArea = np.multiply(densityAnalysisArea.astype(bool).astype(int),
                                          np.linspace(1, 10, np.shape(densityAnalysisArea)[1]))

density = np.count_nonzero(densityAnalysisArea, axis=1)
avgDensity = np.average(density)

weiDensity = np.sum(weightedDensityAnalysisArea, axis=1)
avgWeiDensity = np.average(weiDensity)

print('IMFs:', np.shape(imfsAll))
print('IMFsNorm:', np.shape(imfsAllNorm))
print('IMFsEnv:', np.shape(imfsEnv))
print('InstFs:', np.shape(instfAll))
print('InstFsNorm:', np.shape(instfAllNorm))
print('Magnitudes:', np.shape(magAll))
print('Specs:', np.shape(specs))
print('densityAnalysisArea:', np.shape(densityAnalysisArea))

# for inf in instfAll:
#     inf /= max(inf)  # normalization
# for imf2 in instfAll2:
#     imf2 /= max(imf2)  # normalization
# for imf in imfsAll:
#     imf /= max(imf)  # normalization

print('plotting...')

plotDensity = True
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

plotSpec = False
if plotSpec:
    # Make plot with vertical (default) colorbar
    fig, ax = plt.subplots(figsize=(15, 30))

    data = np.swapaxes(specs, 0, 1)
    data = np.ma.masked_where(data < 0.001, data)

    cmap = plt.cm.magma
    cmap.set_bad(color='white')

    cax = ax.imshow(data, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto')
    ax.set_title(filename + ' Magnitudes of IMFs')

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 2, 4000])
    cbar.ax.set_yticklabels(['0', '> ', '23'])  # vertically oriented colorbar
    # plt.gca().invert_yaxis()
    plt.savefig('plots/graphMag' + filename + '.png', dpi=200)

    plt.show()

plotImfs = False
if plotImfs:
    for i in range(np.shape(imfsAll)[0]):
        # Plot Original Signal
        # plt.plot(np.divide(range(len(sig)), sampRate), sig,
        #          label="Orig", color='gray', linewidth=0.5, zorder=-3)
        # Plot Instant Frequencies
        plt.subplot(211)
        plt.title('IMF #' + str(i+1))
        plt.plot(np.divide(range(len(instfAll[i])), sampRate), instfAll[i],
                 label="Instf", color='green', linewidth=0.7, zorder=2)
        plt.plot(np.divide(range(len(instfAllNorm[i])), sampRate), instfAllNorm[i],
                 label="Instf from NormImfs", color='brown', linewidth=0.7, zorder=5)
        plt.grid(True, linestyle='dashed', color='darkred', linewidth=0.2, zorder=0)
        plt.xticks(np.arange(0, len(sig) / sampRate, step=0.1), rotation=90)
        # plt.ylabel('I')
        plt.legend()
        # Plot Imfs
        plt.subplot(212)
        # plt.title('IMF #' + str(i+1))
        plt.plot(np.divide(range(len(imfsAll[i])), sampRate), np.divide(imfsAll[i], max(imfsAll[i])),
                 label="Imf(normalized for plotting)", color='orange', linewidth=0.75, zorder=2)
        plt.plot(np.divide(range(len(imfsAllNorm[i])), sampRate), imfsAllNorm[i],
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


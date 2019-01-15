"""
Code taken from:
https://pyhht.readthedocs.io/en/latest/apiref/pyhht.html#pyhht.utils.inst_freq
[2018.08.30]: Tried, works as example
"""
from pyhht.emd import EMD
from utils import readMonoWav
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np
np.set_printoptions(edgeitems=50)

# sampRate, sig = readMonoWav('./METU Recordings/hh2_48kHz_Mono_32bitFloat.wav')
# sampRate, sig = readMonoWav('./METU Recordings/hh2_breath/hh2_04_00.34.164_270_en.wav')
# sampRate, sig = readMonoWav('./METU Recordings/hh2_breath/hh2_09_01.20.741_134_en2.wav')
# sampRate, sig = readMonoWav('./METU Recordings/hh2_breath/hh2_09_01.20.741_134_en3_16bit.wav')
# sampRate, sig = readMonoWav('./METU Recordings/hh2_breath/hh2_23_03.02.050_149_tr.wav')
sampRate, sig = readMonoWav('./METU Recordings/hh2_withEdges/hh2_random001.wav')

# print(sig)
print(len(sig))
print(sampRate)
print('######')

readWholeFile = True  # set to False for trials only

if readWholeFile is False:
    # take a sample for better visualising, huge chunks crashes in hilbert()
    sig = sig[sampRate*0:sampRate*1]  # read only first 1 second

# sig = sig / max(sig)  # normalization

imfsAll = None
instfAll = None
partLen = sampRate // 10  # divide into 100 ms parts - shorter durations lead to less imfs
partCount = int(np.ceil(len(sig) / partLen))
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
    imfsPart = list(decomposer.decompose())[:-1]  # last element is residue

    # todo - ai : normalize imfs here to handle negative parts

    # calculate instant frequency for each imf of the part
    instfPart = []
    truncatedImfs = []
    for imf in imfsPart:
        hx = sp.hilbert(imf)
        phx = np.unwrap(np.arctan2(hx.imag, hx.real))
        tempInstf = sampRate / (2 * np.pi) * np.diff(phx)

        # removing neighbor parts after calculations
        if i > 0:  # not first part
            tempInstf = tempInstf[partLen:]
            imf = imf[partLen:]
        if i < partCount - 2:  # until second from last part
            tempInstf = tempInstf[:-partLen]
            imf = imf[:-partLen]
        if i == partCount - 2:  # second from last part (last part's len may not be partLen)
            tempInstf = tempInstf[:-(len(sig) % partLen)]
            imf = imf[:-(len(sig) % partLen)]
        truncatedImfs.append(imf)
        instfPart.append(tempInstf)

    # done with extra parts, set truncated imfs
    imfsPart = truncatedImfs

    # concatanate all parts' imfs together
    if imfsAll is None:
        imfsAll = list(imfsPart)
    else:
        while len(imfsAll) < len(imfsPart):  # if instf of this part has MORE rows
            imfsAll.append(np.zeros(len(imfsAll[0])))  # add new imfs row to main list
        while len(imfsPart) < len(imfsAll):  # if instf of this part has LESS rows
            imfsPart.append(np.zeros(len(imfsPart[0])))  # add new imf row to part
        imfsAll = list(np.concatenate((imfsAll, imfsPart), axis=1))

    # concatanate all parts' instant frequency together
    if instfAll is None:
        instfAll = list(instfPart)
    else:
        while len(instfAll) < len(instfPart):  # if instf of this part has MORE rows
            instfAll.append(np.zeros(len(instfAll[0])))  # add new imfs row to main list
        while len(instfPart) < len(instfAll):  # if instf of this part has LESS rows
            instfPart.append(np.zeros(len(instfPart[0])))  # add new imf row to part
        instfAll = list(np.concatenate((instfAll, instfPart), axis=1))

print('IMFs:', np.shape(imfsAll))
print('InstFs:', np.shape(instfAll))

# for inf in instfAll:
#     inf /= max(inf)  # normalization
# for imf in imfsAll:
#     imf /= max(imf)  # normalization

# todo - ai: plot spectogram of each frame by summing each freq bin in each imf's instf
print('plotting...')

for i in range(np.shape(instfAll)[0]):
    plt.plot(np.divide(range(len(sig)), sampRate), sig,
             label="Orig", color='gray', linewidth=0.5, zorder=-3)
    plt.plot(np.divide(range(len(instfAll[i])), sampRate), instfAll[i],
             label="Instf", color='green', linewidth=0.9, zorder=-2)
    plt.plot(np.divide(range(len(imfsAll[i])), sampRate), imfsAll[i],
             label="Imf", color='orange', linewidth=0.75, zorder=-1)
    plt.grid(True, color='darkred', linewidth=0.2, zorder=0)
    plt.xticks(np.arange(0, len(sig) / sampRate, step=0.1), rotation=90)
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Amplitude')
    plt.title('Instant Freq of IMF #' + str(i+1))
    plt.legend()
    plt.savefig("plots/graphImf"+str(i+1)+".svg")
    plt.savefig("plots/graphImf"+str(i+1)+".png")
    plt.show()


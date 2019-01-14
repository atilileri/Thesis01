"""
Code taken from:
https://pyhht.readthedocs.io/en/latest/apiref/pyhht.html#pyhht.utils.inst_freq
[2018.08.30]: Tried, works as example
"""
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
from tftb.processing import inst_freq
from utils import readMonoWav
from scipy.signal import hilbert
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

instfAll = None
partLen = sampRate // 10  # divide into 100 ms parts - shorter durations lead to less imfs
partCount = int(np.ceil(len(sig) / partLen))
for i in range(partCount):
    startIndex = i*partLen
    endIndex = (i+1)*partLen
    # temporarily adding previous and following parts for more accurate calculations
    if i > 0:  # if not first part
        startIndex -= partLen
    if i < partCount - 2:  # until second from last part
        endIndex += partLen
    if i == partCount - 2:  # second from last part (last part's len may not be partLen)
        endIndex += len(sig) % partLen
    part = sig[startIndex:endIndex]

    decomposer = EMD(part)
    imfsOfThisPart = list(decomposer.decompose())[:-1]  # last element is residue
    instfPart = []
    for imf in imfsOfThisPart:
        hx = sp.hilbert(imf)
        phx = np.unwrap(np.arctan2(hx.imag, hx.real))
        tempInstf = sampRate / (2 * np.pi) * np.diff(phx)

        # removing previous and following parts
        if i > 0:  # not first part
            tempInstf = tempInstf[partLen:]
        if i < partCount - 2:  # until second from last part
            tempInstf = tempInstf[:-partLen]
        if i == partCount - 2:  # second from last part (last part's len may not be partLen)
            tempInstf = tempInstf[:-(len(sig) % partLen)]

        instfPart.append(tempInstf)

    if instfAll is None:
        instfAll = list(instfPart)
    else:
        while len(instfAll) < len(instfPart):  # if instf of this part has MORE rows
            instfAll.append(np.zeros(len(instfAll[0])))  # add new imfs row to main list
        while len(instfPart) < len(instfAll):  # if instf of this part has LESS rows
            instfPart.append(np.zeros(len(instfPart[0])))  # add new imf row to part
        instfAll = list(np.concatenate((instfAll, instfPart), axis=1))

print(np.shape(instfAll))

# for inf in instfAll:
#     inf /= max(inf)  # normalization

print('plotting...')

for i in range(np.shape(instfAll)[0]):
    plt.plot(np.divide(range(len(sig)), sampRate), sig, label="Orig", color='gray', linewidth=0.5)
    plt.plot(np.divide(range(len(instfAll[i])), sampRate), instfAll[i], label="Instf", color='green', linewidth=0.9)
    # plt.plot(np.divide(range(len(imfs[i])), sampRate), imfs[i], label="Imf", color='orange', linewidth=0.75)
    plt.grid(True, linestyle='dotted')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Amplitude')
    plt.title('Instant Freq of IMF #' + str(i+1))
    plt.legend()
    plt.savefig("graphImf"+str(i+1)+".svg")
    plt.show()


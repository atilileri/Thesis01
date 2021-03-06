"""
Multiple Implementations of Power Spectral Entropy
"""
from utils import readMonoWav
import matplotlib.pyplot as plt
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

partLen = sampRate // 100  # divide into 10 ms parts
partCount = int(np.ceil(len(sig) / partLen))
"""
PSE Implementation v1
Inspired from :
https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy
"""
psdArrv1 = []
pseArrv1 = []
for i in range(partCount):
    part = sig[i*partLen:(i+1)*partLen]
    w = np.fft.fft(part)
    wAbs = np.abs(w, dtype=float)
    wSq = np.square(wAbs, dtype=float)
    psd = wSq / len(w)
    if len(psd) < partLen:
        psd = np.append(psd, np.full(partLen-len(psd), 0.00000001))
    psdArrv1.append(psd)
psdSum = np.sum(psdArrv1)
for i in range(partCount):
    normPsd = psdArrv1[i] / psdSum
    pse = -np.sum(normPsd*np.log(normPsd))
    pseArrv1.append(pse)

print('plotting...')
plt.plot(np.divide(range(len(sig)), sampRate), sig,
         label="Orig", color='0.75', linewidth=0.5, zorder=-10)
plt.plot(np.multiply(range(len(pseArrv1)), partLen) / sampRate, pseArrv1,
         label="pse1", color='blue', linewidth=1, zorder=-5)
plt.grid(True, color='darkred', linewidth=0.2, zorder=0)
plt.xticks(np.arange(0, len(sig) / sampRate, step=0.1), rotation=90)
plt.xlabel('Time (Seconds)')
plt.ylabel('Amplitude')
plt.title('Power Spectral Energy')
plt.legend()
plt.savefig("plots/pseV1.svg")
plt.savefig("plots/pseV1.png")
plt.show()



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
import numpy
numpy.set_printoptions(edgeitems=50)

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

readWholeFile = True  # for trials only

if readWholeFile is False:
    # take a sample for better visualising, huge chunks crashes in hilbert()
    sig = sig[sampRate*0:sampRate*1]  # read only first 1 second

instf = []

partLen = sampRate // 10  # divide into 100 ms parts
imfs = []
for i in range(len(sig)//partLen):
    part = sig[i*partLen:(i+1)*partLen]
    decomposer = EMD(part, n_imfs=8)
    partImfs = decomposer.decompose()

    for j in range(len(partImfs) - 1):  # last element is residue
        # Converting imfs into an analytic one using hilbert(), according to:
        # https://github.com/jaidevd/pyhht/issues/43#issuecomment-398077924
        partInstf, _ = inst_freq(hilbert(partImfs[j]))
        if i == 0:
            imfs.append([])
            instf.append([])
        imfs[j].extend(partImfs[j])
        instf[j].extend(partInstf)

    if i == 0:  # todo: remove whole if block after use
        print(numpy.shape(partImfs))
        plot_imfs(part, partImfs)

print(numpy.shape(instf))
print('plotting...')
for i in range(numpy.shape(instf)[0]):
    plt.plot(range(len(sig)), sig, label="Orig", color='gray')
    plt.plot(range(len(instf[i])), instf[i], label="Instf", color='green')
    plt.plot(range(len(imfs[i])), imfs[i], label="Imf", color='orange')
    plt.grid(True, linestyle='dotted')
    plt.xlabel('Time (Samples)')
    plt.ylabel('Amplitude')
    plt.title('Instant Freq of IMF #' + str(i+1))
    plt.legend()
    plt.show()

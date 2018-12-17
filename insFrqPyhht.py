"""
Code taken from:
https://pyhht.readthedocs.io/en/latest/apiref/pyhht.html#pyhht.utils.inst_freq
[2018.08.30]: Tried, works as example
"""
from tftb.processing import inst_freq
from tftb.generators import fmsin
import matplotlib.pyplot as plt
from utils import readMonoWav
import numpy
from scipy.signal import hilbert
numpy.set_printoptions(edgeitems=50)

sig = readMonoWav('./METU Recordings/hh2_48kHz_Mono_32bitFloat.wav')[1]
# sig = readMonoWav('./METU Recordings/hh2_breath/hh2_04_00.34.164_270_en.wav')[1]
# sig = readMonoWav('./METU Recordings/hh2_breath/hh2_09_01.20.741_134_en2.wav')[1]
# sig = readMonoWav('./METU Recordings/hh2_breath/hh2_09_01.20.741_134_en3_16bit.wav')[1]
# sig = readMonoWav('./METU Recordings/hh2_breath/hh2_23_03.02.050_149_tr.wav')[1]

print(sig, len(sig))
print('##########################################################################')
# print(type(sig[1]))
# x = fmsin(70, 0.05, 0.35, 25)[0]
# print(type(sig))

readWholeFile = True

if readWholeFile:  # for tests if file is readable. Does not visualize well and plotting takes too long
    # todo - hh: may be we have to calculate average on small chunks?
    timestamps = []
    instf = []

    for i in range(len(sig)//1000):
        part = sig[i*1000:(i+1)*1000]
        # Converting signal into an analytic one using hilbert(), according to:
        # https://github.com/jaidevd/pyhht/issues/43#issuecomment-398077924
        part = hilbert(part)
        partInstf, partTimestamps = inst_freq(part)
        instf.extend(partInstf)
        timestamps.extend(numpy.add(partTimestamps, i*1000))
        print(i)

else:
    # take a sample for better visualising, huge chunks crashes in hilbert()
    sig = sig[:3000]

    # Converting signal into an analytic one using hilbert(), according to:
    # https://github.com/jaidevd/pyhht/issues/43#issuecomment-398077924
    sig = hilbert(sig)

    instf, timestamps = inst_freq(sig)

print('calc is over')
# print(instf)
print(numpy.shape(instf))
print('plotting...')

plt.plot(timestamps, instf)
plt.grid(True, linestyle='dotted')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency')
plt.title('Instantaneous frequency estimation')
plt.show()


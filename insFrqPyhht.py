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
print(sig, len(sig))
print('##########################################################################')
# print(type(sig[1]))
# x = fmsin(70, 0.05, 0.35, 25)[0]
# print(type(sig))

# take a sample for better visualising, huge chunks crashes in hilbert()
sig = sig[4000000:4001000]

# Changes according to
# https://github.com/jaidevd/pyhht/issues/43#issuecomment-398077924
sig = hilbert(sig)


instf, timestamps = inst_freq(sig)
print(instf, len(instf))

# todo - atili: try more with whole_speech and hh2 wave files

plt.plot(timestamps, instf)
plt.grid(True, linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Instantaneous frequency estimation')
plt.show()


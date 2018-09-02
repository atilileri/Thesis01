"""
Code taken from:
https://pytftb.readthedocs.io/en/master/nonstationary_signals.html#instantaneous-frequency
[2018.08.30]: Tried, works as example
"""
import numpy as np
from tftb.generators import fmlin
from tftb.processing import inst_freq, plotifl
from utils import readMonoWav
import numpy
numpy.set_printoptions(edgeitems=50)

signal = readMonoWav('./METU Recordings/hh2_48kHz_Mono_32bitFloat.wav')[1]
print(signal, len(signal))
print('##########################################################################')
# signal, _ = fmlin(256)
# time_samples = np.arange(3, 257)

ifr = inst_freq(signal)[0]
print(ifr, len(ifr))

# todo - atili: try more with whole_speech and hh2 wave files
# plotifl(time_samples, ifr)

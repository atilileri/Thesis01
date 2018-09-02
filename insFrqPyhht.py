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
numpy.set_printoptions(edgeitems=50)

x = readMonoWav('./METU Recordings/hh2_48kHz_Mono_32bitFloat.wav')[1]
print(x, len(x))
print('##########################################################################')
# print(type(x[1]))
# x = fmsin(70, 0.05, 0.35, 25)[0]
# print(type(x))
instf, timestamps = inst_freq(x)
print(instf, len(instf))

# todo - atili: try more with whole_speech and hh2 wave files
'''
plt.plot(timestamps, instf)
plt.grid(True, linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Instantaneous frequency estimation')
plt.show()
'''

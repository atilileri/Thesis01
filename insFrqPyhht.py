"""
Code taken from:
https://pyhht.readthedocs.io/en/latest/apiref/pyhht.html#pyhht.utils.inst_freq
[2018.08.30]: Tried, works as example
"""

from tftb.processing import inst_freq
from tftb.generators import fmsin
import matplotlib.pyplot as plt

x = fmsin(70, 0.05, 0.35, 25)[0]
instf, timestamps = inst_freq(x)
plt.plot(timestamps, instf)
plt.grid(True, linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Instantaneous frequency estimation')
plt.show()

# todo - atili: try with whole_speech and hh2 wave files

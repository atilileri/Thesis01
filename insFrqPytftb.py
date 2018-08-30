"""
Code taken from:
https://pytftb.readthedocs.io/en/master/nonstationary_signals.html#instantaneous-frequency
[2018.08.30]: Tried, works as example
"""
import numpy as np
from tftb.generators import fmlin
from tftb.processing import inst_freq, plotifl

signal, _ = fmlin(256)
time_samples = np.arange(3, 257)
ifr = inst_freq(signal)[0]
plotifl(time_samples, ifr)

# todo - atili: try with whole_speech and hh2 wave files

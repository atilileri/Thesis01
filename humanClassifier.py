import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.io.wavfile
import scipy.signal
import sounddevice as sd
import numpy as np
import os
from datetime import datetime
import shutil
import sys

filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10/'
# filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/Recordings_Max/'
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

for root, directories, files in os.walk(filepath):
    fileIdx = 0
    while fileIdx < len(files):
        fullPath = root + files[fileIdx]
        print('\nListening Breath #', fileIdx+1, 'of', len(files), ':', files[fileIdx], 'at', root, flush=True)
        samplingRate, audio = scipy.io.wavfile.read(fullPath)
        # for chIdx in range(len(audio[0])):
        #     sd.play(audio[:, chIdx], samplingRate, blocking=True)
        sd.play(audio[:, 0:2], samplingRate, blocking=False)
        # sd.play(audio[:, 2:4], samplingRate, blocking=False)

        newFolder = ''
        answer = input('[Enter,R]epeat - [1]Primary Breath - [2]Secondary Breath '
                       '- [0,N]ot Breath - [U]ndecided - [Q]uit - [P]ass: ')
        if 'r' == answer or 'R' == answer or '' == answer:
            print('Repeat')
        elif '1' == answer:
            print('Primary Breath')
            newFolder = '_1/'
        elif '2' == answer:
            print('Secondary Breath')
            newFolder = '_2/'
        elif 'n' == answer or 'N' == answer or '0' == answer:
            print('Not Breath')
            newFolder = '_NotBreath/'
        elif 'u' == answer or 'U' == answer:
            print('Undecided')
            newFolder = '_Undecided/'
        elif 'q' == answer or 'Q' == answer:
            print('Quit')
            sys.exit()
        elif 'p' == answer or 'P' == answer:
            print('Pass')
            fileIdx += 1
        else:
            print('Wrong input. Repeating.')

        if '' != newFolder:
            newFolder = root.strip('/') + newFolder
            if not os.path.exists(newFolder):
                os.makedirs(newFolder)
            shutil.move(fullPath, newFolder+files[fileIdx])
            fileIdx += 1

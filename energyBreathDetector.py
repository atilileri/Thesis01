import utils
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice as sd
import time
import numpy as np
import os
from datetime import datetime


def splitBreaths(path, name, timestamp, verbose):
    fs, inputSignal = scipy.io.wavfile.read(path+'/'+name)

    windowLengthInSeconds = 0.001
    minLenInSeconds = 0.2
    maxLenInSeconds = 1.0
    minLenInFrames = int(minLenInSeconds * fs)
    maxLenInFrames = int(maxLenInSeconds * fs)

    breathStartMaxEnergy = -40
    breathEndMinEnergy = -25
    startEdgeMaxEnergy = -50
    stopEdgeMaxEnergy = -55
    inputSignal = inputSignal[:, 0]
    # todo - ai : cut all channels according to first

    windowLengthInFrames = int(fs*windowLengthInSeconds)
    energies = []
    zcrs = []
    for i in range(0, len(inputSignal), windowLengthInFrames):
        energies.append(utils.calcShortTimeEnergy(inputSignal[i:i+windowLengthInFrames])[1])
        # todo - ai : remove zcr if unnecessary
        zcrs.append(0)  # utils.calcZeroCrossingRate(inputSignal[i:i+windowLengthInFrames]))

    breaths = []
    breathing = False
    startWindowIndex = 0
    stopWindowIndex = 0
    # First(Detect) Energy Classification
    for i in range(len(energies)):
        if energies[i] < breathStartMaxEnergy and breathing is False:
            startWindowIndex = i
            breathing = True
        elif energies[i] > breathEndMinEnergy and breathing is True:
            stopWindowIndex = i
            # Second(Trim) Energy Classification
            if stopWindowIndex - startWindowIndex > 10:  # pseudo minimum for too short windows
                midWindowIndex = (startWindowIndex + stopWindowIndex) // 2
                half = energies[startWindowIndex:midWindowIndex]
                minEngStart = min(half)
                offset = half.index(minEngStart)
                startWindowIndex += offset
                half = energies[midWindowIndex:stopWindowIndex]
                minEngStop = min(half)
                offset = half.index(minEngStop)
                stopWindowIndex = midWindowIndex + offset + 1

                startFrame = int(startWindowIndex * windowLengthInFrames)
                stopFrame = int(stopWindowIndex * windowLengthInFrames)
                # Duration Classification and Third(Edge) Energy Classification
                if (minLenInFrames < stopFrame - startFrame < maxLenInFrames)\
                        and minEngStart < startEdgeMaxEnergy and minEngStop < stopEdgeMaxEnergy:
                    breaths.append((startFrame, stopFrame,
                                    zcrs[startWindowIndex:stopWindowIndex],
                                    energies[startWindowIndex:stopWindowIndex]))
            breathing = False
            startWindowIndex = 0
            stopWindowIndex = 0

    # save, print and play
    breathIdx = 1
    for s in breaths:
        sig = inputSignal[s[0]:s[1]]
        LenInFrames = s[1] - s[0]
        if verbose:
            plt.figure(figsize=(10, 10))
            # SIGNAL
            plt.subplot(2, 2, 1)
            plt.title('Signal')
            plt.plot(sig)
            # ENERGY
            plt.subplot(2, 2, 2)
            plt.title('Short Time Energy')
            plt.plot(s[3])
            # ZCR
            plt.subplot(2, 2, 3)
            plt.title('Zero Crossing Rate')
            plt.plot(s[2])
            # File Information and Consideration Parameters
            plt.subplot(2, 2, 4)
            plt.title('File Info')
            plt.axis([0, 10, 0, 10])
            t = ('Record: ' + name + ' Breath #' + str(breathIdx) + '\n' +
                 'Time(sec): ' + str(s[0]/fs) + '-' + str(s[1]/fs) + ' (' + str(LenInFrames/fs) + ')\n\n' +
                 'Signal Mean: ' + str(np.mean(sig)) + '\n' +
                 'Signal Max: ' + str(np.max(sig)) + '\n' +
                 'Signal Min: ' + str(np.min(sig)) + '\n' +
                 'Signal Var: ' + str(np.var(sig)) + '\n' +
                 'Signal StDev: ' + str(np.std(sig)) + '\n\n' +
                 'Energy Mean: ' + str(np.mean(s[3])) + '\n' +
                 'Energy Max: ' + str(np.max(s[3])) + '\n' +
                 'Energy Min: ' + str(np.min(s[3])) + '\n' +
                 'Energy Var: ' + str(np.var(s[3])) + '\n' +
                 'Energy StDev: ' + str(np.std(s[3])) + '\n\n' +
                 'ZCR Mean: ' + str(np.mean(s[2])) + '\n' +
                 'ZCR Max: ' + str(np.max(s[2])) + '\n' +
                 'ZCR Min: ' + str(np.min(s[2])) + '\n' +
                 'ZCR Var: ' + str(np.var(s[2])) + '\n' +
                 'ZCR StDev: ' + str(np.std(s[2])) + '\n\n')
            plt.text(0.5, -0.75, t, wrap=True, fontsize=13)
            plt.tight_layout()
            fig = plt.gcf()
            plt.show()
            sd.play(sig, fs)
            time.sleep(LenInFrames/fs)
            sd.stop()
            print(str(breathIdx)+'-', s[0]/fs, '-', s[1]/fs, LenInFrames/fs)
            # for manual classification and waits
            # classification = input('classify:')
            # fig.savefig('./plots/breathClassification/'+classification
            #             + '_'+filename+'_fig'+str(breathIdx)+'.png')

        # save file
        namepieces = name.split('.')
        savefilename = os.path.dirname(path)+'/breaths_'+timestamp+'/'+namepieces[0]+'_'+str(breathIdx)+'_'+str(s[0]/fs)+'-'+str(LenInFrames/fs)+'.'+namepieces[1]
        if not os.path.exists(os.path.dirname(savefilename)):
            os.makedirs(os.path.dirname(savefilename))

        # print(savefilename)
        scipy.io.wavfile.write(savefilename, fs, sig)

        breathIdx = breathIdx + 1


filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/'
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
verboseOutput = False

for root, directories, files in os.walk(filepath):
    for file in files:
        if '.wav' in file:
            print('Extracting Breaths of:', file, 'at', root)
            splitBreaths(root, file, ts, verboseOutput)

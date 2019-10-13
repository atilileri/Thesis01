import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.io.wavfile
import scipy.signal
import sounddevice as sd
import numpy as np
import os
from datetime import datetime


def splitBreaths(path, name, timestamp, verboseSignal, verboseEnergy, playAudio, savePlots, saveFiles, printDelays):
    fs, inputAllChannels = scipy.io.wavfile.read(path+'/'+name)

    windowLengthInSeconds = 0.00050
    windowLengthInSamples = int(fs*windowLengthInSeconds)

    minLenInSeconds = 0.2
    maxLenInSeconds = 1.0
    minLenInSamples = int(minLenInSeconds * fs)
    maxLenInSamples = int(maxLenInSeconds * fs)

    breathStartMaxEnergy = -40
    breathEndMinEnergy = -25
    inputAllChannels = np.swapaxes(inputAllChannels, 0, 1)

    dataInfo = []
    breathSections = []
    channelInfo = dict()
    # calculate parameters
    print('Calculating energies for windows...')
    for chIdx in range(len(inputAllChannels)):  # do for first channel only
        channelInfo['audioSamples'] = inputAllChannels[chIdx]
        channelInfo['energyOfWindows'] = list()
        if (0 == chIdx) or (verboseEnergy is True):  # do not calculate chnl 2,3,4 if not displayed
            for windowStartSampleIdx in range(0, len(channelInfo['audioSamples']), windowLengthInSamples):
                channelInfo['energyOfWindows'].append(utils.calcShortTimeEnergy(
                    channelInfo['audioSamples'][windowStartSampleIdx:windowStartSampleIdx+windowLengthInSamples])[1])
                # channelInfo['zcr'].append(utils.calcZeroCrossingRate(
                #     channelInfo['audio'][windowStartSampleIdx:windowStartSampleIdx+windowLengthInSamples]))
        dataInfo.append(channelInfo.copy())
        channelInfo.clear()
    del channelInfo

    print('Applying classifications...')
    breathing = False
    startWindowIndex = 0
    # First(Detect) Energy Classification
    breathSectionInfo = dict()
    for windowIdx in range(len(dataInfo[0]['energyOfWindows'])):
        if dataInfo[0]['energyOfWindows'][windowIdx] < breathStartMaxEnergy and breathing is False:
            startWindowIndex = windowIdx
            breathing = True
        elif dataInfo[0]['energyOfWindows'][windowIdx] > breathEndMinEnergy and breathing is True:
            stopWindowIndex = windowIdx
            # Second(Trim) Energy Classification
            if stopWindowIndex - startWindowIndex > 10:  # pseudo minimum for too short windows
                startWindowIndexBeforeTrim = startWindowIndex
                stopWindowIndexBeforeTrim = stopWindowIndex
                midWindowIndex = (startWindowIndex + stopWindowIndex) // 2
                half = dataInfo[0]['energyOfWindows'][startWindowIndex:midWindowIndex]
                minEngStart = min(half)
                offset = half.index(minEngStart)
                startWindowIndex += offset
                half = dataInfo[0]['energyOfWindows'][midWindowIndex:stopWindowIndex]
                minEngStop = min(half)
                offset = half.index(minEngStop)
                stopWindowIndex = midWindowIndex + offset + 1

                startSample = int(startWindowIndex * windowLengthInSamples)
                startSampleBT = int(startWindowIndexBeforeTrim * windowLengthInSamples)
                stopSample = int(stopWindowIndex * windowLengthInSamples)
                stopSampleBT = int(stopWindowIndexBeforeTrim * windowLengthInSamples)
                # Duration Classification and Third(Edge) Energy Classification
                if minLenInSamples < (stopSample - startSample) < maxLenInSamples:
                    breathSectionInfo['startSample'] = startSample
                    breathSectionInfo['stopSample'] = stopSample
                    breathSectionInfo['startSampleBT'] = startSampleBT
                    breathSectionInfo['stopSampleBT'] = stopSampleBT
                    breathSectionInfo['startWindow'] = startWindowIndex
                    breathSectionInfo['stopWindow'] = stopWindowIndex
                    breathSectionInfo['startWindowBT'] = startWindowIndexBeforeTrim
                    breathSectionInfo['stopWindowBT'] = stopWindowIndexBeforeTrim
                    breathSectionInfo['offsets'] = list()
                    breathSections.append(breathSectionInfo.copy())
                    breathSectionInfo.clear()
            breathing = False
            startWindowIndex = 0
    del breathSectionInfo

    # save, print and play
    plotSavePath = './plots/' + timestamp + '/'

    for sectionIdx in range(len(breathSections)):
        namepieces = name.split('.')
        fileSaveName = (namepieces[0] + '_' + '{:03d}'.format(sectionIdx + 1))

        for chIdx in range(len(dataInfo)):
            # calculate offsets for propagation delay (by cross correlation)
            breathFound = (dataInfo[0]['audioSamples'][breathSections[sectionIdx]['startSample']:
                                                       breathSections[sectionIdx]['stopSample']])
            examineSection = (dataInfo[chIdx]['audioSamples'][breathSections[sectionIdx]['startSample']:
                                                              breathSections[sectionIdx]['stopSampleBT']])
            corr = np.correlate(examineSection, breathFound, 'valid')

            offset = np.abs(corr).argmax()
            breathSections[sectionIdx]['offsets'].append(offset)
            if printDelays:
                print('Delay for channel #%d: %d samples' % (chIdx, offset))

        if verboseSignal:
            print('Plotting Signal...')
            plt.figure(figsize=(10, 10))
            for chIdx in range(len(dataInfo)):
                # draw
                plt.subplot(2, 2, chIdx+1)
                plt.title('Signal Channel '+str(chIdx+1))
                # audio samples
                plt.plot(dataInfo[chIdx]['audioSamples']
                         [breathSections[sectionIdx]['startSampleBT']:breathSections[sectionIdx]['stopSampleBT']],
                         linewidth=0.4)
                # trim start line
                plt.axvline(x=(breathSections[sectionIdx]['startSample'] - breathSections[sectionIdx]['startSampleBT']
                               + breathSections[sectionIdx]['offsets'][chIdx]),
                            color='red', linestyle='dashed', zorder=5, linewidth=0.8)
                # trim end line
                plt.axvline(x=(breathSections[sectionIdx]['stopSample'] - breathSections[sectionIdx]['startSampleBT']
                               + breathSections[sectionIdx]['offsets'][chIdx]),
                            color='red', linestyle='dashed', zorder=5, linewidth=0.8)
            # create legend
            plt.figlegend(handles=[
                Line2D([0], [0], color='#1f77b4', label='Audio'),
                Line2D([0], [0], color='red', linestyle='dashed', label='Trim Lines(synced)')
            ], loc='center')

            plt.tight_layout()
            # hold on to figure for manual classification saving
            fig = plt.gcf()
            plt.show()
            if savePlots:
                if not os.path.exists(plotSavePath):
                    os.makedirs(plotSavePath)
                fig.savefig(plotSavePath + fileSaveName + '_Signal.png')
                fig.savefig(plotSavePath + fileSaveName + '_Signal.svg')

        if verboseEnergy:
            print('Plotting Energies...')
            plt.figure(figsize=(10, 10))
            for chIdx in range(len(dataInfo)):
                # draw
                plt.subplot(2, 2, chIdx+1)
                plt.title('Energy Channel '+str(chIdx+1))
                # Energy data
                plt.plot(dataInfo[chIdx]['energyOfWindows']
                         [breathSections[sectionIdx]['startWindowBT']:breathSections[sectionIdx]['stopWindowBT']],
                         color='#1f77b4', linewidth=0.4)
                # trim start line
                plt.axvline(x=(breathSections[sectionIdx]['startWindow'] - breathSections[sectionIdx]['startWindowBT']
                               + np.floor(breathSections[sectionIdx]['offsets'][chIdx] / windowLengthInSamples)),
                            color='red', linestyle='dashed', zorder=5, linewidth=0.8)
                # trim end line
                plt.axvline(x=(breathSections[sectionIdx]['stopWindow'] - breathSections[sectionIdx]['startWindowBT']
                               + np.ceil(breathSections[sectionIdx]['offsets'][chIdx] / windowLengthInSamples)),
                            color='red', linestyle='dashed', zorder=5, linewidth=0.8)
            # create legend
            plt.figlegend(handles=[
                Line2D([0], [0], color='#1f77b4', label='Energy'),
                Line2D([0], [0], color='red', linestyle='dashed', label='Trim Lines(synced)')
            ], loc='center')

            plt.tight_layout()
            # hold on to figure for manual classification saving
            fig = plt.gcf()
            plt.show()
            if savePlots:
                if not os.path.exists(plotSavePath):
                    os.makedirs(plotSavePath)
                fig.savefig(plotSavePath + fileSaveName + '_Energy.png')
                fig.savefig(plotSavePath + fileSaveName + '_Energy.svg')

        if playAudio:
            while 'p' == input('Press p to play, ENTER to skip:'):
                print('Breath #' + str(sectionIdx+1) + ':',
                      breathSections[sectionIdx]['startSample'] / fs, '-',
                      breathSections[sectionIdx]['stopSample'] / fs,
                      (breathSections[sectionIdx]['stopSample'] - breathSections[sectionIdx]['startSample']) / fs,
                      'Playing Audio...')
                for chIdx in range(len(dataInfo)):
                    print('Playing Channel '+str(chIdx+1))
                    sd.play(dataInfo[chIdx]['audioSamples']
                            [breathSections[sectionIdx]['startSample'] + breathSections[sectionIdx]['offsets'][chIdx]
                             :breathSections[sectionIdx]['stopSample'] + breathSections[sectionIdx]['offsets'][chIdx]],
                            samplerate=48000, blocking=True)
        if saveFiles:
            # save file
            fileSavePath = os.path.dirname(os.path.dirname(path)) + '/breaths_'+timestamp+'/'
            fileSaveName += ('_' + '{:08.4f}'.format(breathSections[sectionIdx]['startSample'] / fs) + '-'
                             + '{:06.4f}'.format((breathSections[sectionIdx]['stopSample'] -
                                                  breathSections[sectionIdx]['startSample']) / fs)
                             + '.'+namepieces[-1])  # adding wav extension
            if not os.path.exists(fileSavePath):
                os.makedirs(fileSavePath)

            print('Saving: '+fileSaveName+' to '+fileSavePath, flush=True)
            saveData = list()
            ch = 0
            for chanelData in dataInfo:
                saveData.append(chanelData['audioSamples']
                                [breathSections[sectionIdx]['startSample'] + breathSections[sectionIdx]['offsets'][ch]:
                                 breathSections[sectionIdx]['stopSample'] + breathSections[sectionIdx]['offsets'][ch]])
                ch += 1
            # restore channel index order
            saveData = np.swapaxes(saveData, 0, 1)

            scipy.io.wavfile.write(fileSavePath+fileSaveName, fs, saveData)

        if verboseEnergy or verboseSignal:
            input('Continue?:')


filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/Recordings_Max/'
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

for root, directories, files in os.walk(filepath):
    for file in files:
        if '.wav' in file:
            print('Extracting Breaths of:', file, 'at', root)
            splitBreaths(path=root, name=file, timestamp=ts,
                         verboseSignal=False,
                         verboseEnergy=False,
                         playAudio=False,
                         savePlots=False,
                         saveFiles=True,
                         printDelays=False)

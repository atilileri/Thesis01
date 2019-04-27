import utils
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice as sd
import time

fs, inputSignal = scipy.io.wavfile.read('./by01.wav')

inputSignal = inputSignal[:, 0]

# plt.plot(inputSignal)
windowLengthInSeconds = 0.001
windowLengthInFrames = int(fs*windowLengthInSeconds)
energies = []
for i in range(0, len(inputSignal), windowLengthInFrames):
    energies.append(utils.calcShortTimeEnergy(inputSignal[i:i+windowLengthInFrames])[1])
# print(energies)
plt.plot(energies)
plt.show()

breaths = []
breathing = False
startWindowIndex = 0
stopWindowIndex = 0
for i in range(len(energies)):
    if energies[i] < -45 and breathing is False:
        startWindowIndex = i
        breathing = True
    elif energies[i] > -35 and breathing is True:
        stopWindowIndex = i
        start = int(startWindowIndex*windowLengthInFrames)
        stop = int(stopWindowIndex*windowLengthInFrames)
        # print(start, stop)
        breaths.append([inputSignal[start:stop], start/fs, stop/fs])
        breathing = False
        startWindowIndex = 0
        stopWindowIndex = 0
breathIdx = 1
for s in breaths:
    minLenInSeconds = 0.2
    maxLenInSeconds = 0.5
    minLenInFrames = int(minLenInSeconds*fs)
    maxLenInFrames = int(maxLenInSeconds*fs)
    if minLenInFrames < len(s[0]) < maxLenInFrames:
        print(breathIdx, len(s[0]), len(s[0])/fs, s[1], s[2])
        breathIdx = breathIdx + 1
        sd.play(s[0], fs)
        time.sleep(len(s[0])/fs)
        sd.stop()

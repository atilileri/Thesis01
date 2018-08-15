"""
utils for common use
"""
import scipy.io.wavfile
import numpy as np
from scipy import signal
import python_speech_features
import os
import glob

'''
Project Configuration
'''
# for hht versions
signalType = 1  # 0 for original, 1 for our signal

# for sound file sources
insideProject = True  # if sound files are taken from inside the project folder
if insideProject:
    prefix = '.'
else:
    prefix = 'D:/atili/MMIExt/Audacity'


# function to read wav files, assuring that resulting signal is mono channel.
def readMonoWav(pathToFile):
    samplingRate, samples = scipy.io.wavfile.read(pathToFile)
    # stereo to mono
    if len(samples.shape) > 1 and 1 < samples.shape[1]:  # python evals expressions LTR so it's safe to write this
        # maybe all channels can be used as different inputs to augment the data. Now using only first channel
        # Sample mean convert code
        # samplesMono = np.mean(samples, axis=1, dtype=samples.dtype)
        # Sample first channel code
        samplesMono = samples[:, 0]
    else:
        samplesMono = samples
    return samplingRate, samplesMono


# reads wav files in the given folder
def readExampleSet(folderPath):
    exampleSet = []
    minlen = 999999  # pseudo big number
    '''
    Step II-A.1:
    Several signals containing isolated breath examples are selected, forming the example set. From each example, a section
    of fixed length, typically equal to the length of the shortest example in the set (about 100–160 ms), is derived.
    This length is used throughout the algorithm as the frame length (see Section II-C).
    '''
    for filePath in glob.glob(os.path.join(folderPath, '*.wav')):
        sampleRate, samples = readMonoWav(filePath)

        exampleSet.append([filePath, sampleRate, samples])
        if len(samples) < minlen:
            minlen = len(samples)
            # todo - ai: minlen diff looks huge, create better samples
            # print('min frame:', minlen)

    # fix frame length. Refer to Step B.1
    for file in exampleSet:
        file[2] = file[2][:minlen]

    return exampleSet, minlen/sampleRate


# function to create mfcc  matrix of a audio file
def createMfccMatrix(sig,
                     sampRate,
                     subframeDuration=0.010,  # 10 ms
                     hopSize=0.005):  # 5 ms
    subframeDurationBySample = int(subframeDuration * sampRate)
    alpha = 0.95

    ''' Step II-A.2:
    Each breath example is divided into short consecutive subframes, with duration of 10 ms and hop size of 5 ms.
    Each subframe is then pre-emphasized using a first-order difference filter ( H(z) = 1 - alpha * z^-1 
    where alpha = 0.95)
    '''
    mfccMatrix = list()
    for i in range(0, len(sig), int(hopSize * sampRate)):
        # Index to stop: At the end of the subframe, index must stop before the file ends.
        # This assignment adjusts the last step size according to info above.
        stopIdx = min(i + subframeDurationBySample, len(sig) - 1)
        # print(i)
        # print(stopIdx)
        # todo - hh: ask if lfilter() is implemented right?
        emphasized = signal.lfilter([1, -1 * alpha], [1., 1.], sig[i:stopIdx])

        ''' Step II-A.3:
        For each breath example, the MFCC are computed for every subframe, thus forming a short-time cepstrogram
        representation of the example. The cepstrogram is defined as a matrix whose columns are the MFCC vectors
        for each subframe. Each such matrix is denoted by M(i), i=1,2,...,N where N is the number of examples in the
        examples set. The construction of the cepstrogram is demonstrated in Fig. 4.
        '''
        # http://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.mfcc
        # mfcc() function is designed to work on longer signals and divide them into subframes using winlen and winstep
        # parameters. What we do instead is, we split the sgnal ourselves into subframes, apply the filter on the
        # subframe and put it into mfcc() function. So we get 1 mfcc row for each subframe. Because function is designed
        # to work on longer signals, it returns 2d array, each row holding 1 mfcc vector. Since we only get 1 vector,
        # our result is 2d array with the shape of (1,13). So we only get 1st row, as 1d array of 13 elements.
        mfccMatrix.append(python_speech_features.mfcc(signal=emphasized,
                                                      samplerate=sampRate,
                                                      winlen=subframeDuration,
                                                      winstep=hopSize)[0])  # read above explanation for '[0]' index.
        # todo - hh: ask above design choice and/or implementation is true. we can apply the filter on the whole signal
        # todo - hh: and run mfcc() on the whole signal. Instead we do subframing first and then doing both on same time
        # print(len(mfccMatrix))
        # print(mfccMatrix)

    ''' Step II-A.4:
    For each column of the cepstrogram, DC removal is performed, resulting in the matrix M(i) i=1,2,...,N.
    '''
    # print('Before:', mfccMatrix)
    for i in range(len(mfccMatrix)):
        # print(np.mean(mfccMatrix[i]))
        # print(mfccMatrix[i].shape)
        mfccMatrix[i] = mfccMatrix[i] - np.mean(mfccMatrix[i])
        # print(mfccMatrix[i])

    # print('After:', mfccMatrix)
    return mfccMatrix


def getCenterWindow(frame, windowLength):
    midIndex = len(frame) // 2
    halfWindowLen = windowLength / 2
    if midIndex >= halfWindowLen:
        # we cast to int below, instead of above(where halfWindowLen is defined)
        # because otherwise our slice is one element short, for odd windowLength values
        centerWindow = frame[int(midIndex-halfWindowLen):int(midIndex+halfWindowLen)]
    else:
        centerWindow = None
    return centerWindow


# Short Time Energy is computed over a window located around the center of the frame
def calcShortTimeEnergy(analFrame, winLenInSamples):
    window = getCenterWindow(analFrame, winLenInSamples)
    return np.sum(np.square(window)) / len(window)

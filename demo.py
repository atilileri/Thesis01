import scipy.io.wavfile
import os
import glob
import python_speech_features
import numpy as np

path = 'D:/atili/MMIExt/Audacity/Initial Breath Examples/'
subframeDuration = 0.010  # 10 ms
hopSize = 0.005  # 5 ms


# reads wav files in the given folder
def readExampleSet(folderPath):
    exampleSet = []
    minlen = 999999 # pseudo big number

    for filePath in glob.glob(os.path.join(folderPath, '*.wav')):
        sampleRate, samples = scipy.io.wavfile.read(filePath)
        exampleSet.append([filePath, sampleRate, samples])
        if len(samples) < minlen:
            minlen = len(samples)

    # fix frame length. Refer to Step B.1
    for file in exampleSet:
        file[2] = file[2][:minlen]

    return exampleSet


''' Step B.1:
Several signals containing isolated breath examples are selected, forming the example set. From each example, a section
of fixed length, typically equal to the length of the shortest example in the set (about 100–160 ms), is derived.
This length is used throughout the algorithm as the frame length (see Section II-C).
'''
for fileInfo in readExampleSet(path):
    # print(fileInfo)
    # print(len(fileInfo[2]))

    # todo : implement here. ask if lfilter() does this?
    ''' Step B.2:
    Each breath example is divided into short consecutive subframes, with duration of 10 ms and hop size of 5 ms.
    Each subframe is then pre-emphasized using a first-order difference filter
    '''

    ''' Step B.3:
    For each breath example, the MFCC are computed for every subframe, thus forming a short-time cepstrogram
    representation of the example. The cepstrogram is defined as a matrix whose columns are the MFCC vectors
    for each subframe. Each such matrix is denoted by M(i), i=1,2,...,N where N is the number of examples in the
    examples set. The construction of the cepstrogram is demonstrated in Fig. 4.
    '''
    # http://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.mfcc
    mfccMatrix = python_speech_features.mfcc(signal=fileInfo[2],
                                             samplerate=fileInfo[1],
                                             winlen=subframeDuration,
                                             winstep=hopSize)
    # print(len(mfccMatrix))
    # print(mfccMatrix)

    ''' Step B.4:
    For each column of the cepstrogram, DC removal is performed, resulting in the matrix M(i) i=1,2,...,N.
    '''
    for column in mfccMatrix:
        # print(np.mean(column))
        # print(column)
        column = column - np.mean(column)
        # print(column)

    # print(mfccMatrix)

    ''' Step B.5:
    A mean cepstrogram is computed by averaging the matrices of the example set.
    T = 1/N * Epsilon( M(i) i=1,2,...,N )
    This defines the template matrix T. In a similar manner, a variance matrix V is computed, where the distribution of
    each coefficient is measured along the example set.
    '''
    # todo : implement here. Variance Matrix ?



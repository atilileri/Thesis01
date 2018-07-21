import scipy.io.wavfile
import os
import glob
import python_speech_features
import numpy as np
from scipy import signal

path = 'D:/atili/MMIExt/Audacity/Initial Breath Examples/'
subframeDuration = 0.010  # 10 ms
hopSize = 0.005  # 5 ms
alpha = 0.95


# reads wav files in the given folder
def readExampleSet(folderPath):
    exampleSet = []
    minlen = 999999  # pseudo big number

    for filePath in glob.glob(os.path.join(folderPath, '*.wav')):
        sampleRate, samples = scipy.io.wavfile.read(filePath)
        # stereo to mono
        if 1 < samples.shape[1]:
            # todo - hh: maybe all channels can be used as different inputs to augment the data?
            samplesMono = np.mean(samples, axis=1, dtype=int)
        else:
            samplesMono = samples
        exampleSet.append([filePath, sampleRate, samplesMono])
        if len(samplesMono) < minlen:
            minlen = len(samplesMono)
            # todo - ai: minlen diff looks huge, create better samples
            print('min frame:', minlen)

    # fix frame length. Refer to Step B.1
    for file in exampleSet:
        file[2] = file[2][:minlen]

    return exampleSet


''' Step II-A.1:
Several signals containing isolated breath examples are selected, forming the example set. From each example, a section
of fixed length, typically equal to the length of the shortest example in the set (about 100–160 ms), is derived.
This length is used throughout the algorithm as the frame length (see Section II-C).
'''
allMatrixesOfExampleSet = list()
for fileInfo in readExampleSet(path):
    # print(fileInfo)
    # print(len(fileInfo[2]))
    subframeDurationBySample = int(subframeDuration * fileInfo[1])

    # todo - hh: ask if lfilter() works true?
    ''' Step II-A.2:
    Each breath example is divided into short consecutive subframes, with duration of 10 ms and hop size of 5 ms.
    Each subframe is then pre-emphasized using a first-order difference filter ( H(z) = 1 - alpha * z^-1 
    where alpha = 0.95)
    '''
    mfccMatrix = list()
    for i in range(0, len(fileInfo[2]), int(hopSize * fileInfo[1])):
        # Index to stop: At the end of the subframe, index must stop before the file ends.
        # This assignment adjusts the last step size according to info above.
        stopIdx = min(i + subframeDurationBySample, len(fileInfo[2])-1)
        # print(i)
        # print(stopIdx)
        emphasized = signal.lfilter([1, -1*alpha], -1, fileInfo[2][i:stopIdx])

        ''' Step II-A.3:
        For each breath example, the MFCC are computed for every subframe, thus forming a short-time cepstrogram
        representation of the example. The cepstrogram is defined as a matrix whose columns are the MFCC vectors
        for each subframe. Each such matrix is denoted by M(i), i=1,2,...,N where N is the number of examples in the
        examples set. The construction of the cepstrogram is demonstrated in Fig. 4.
        '''
        # http://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.mfcc
        # mfcc() function is designed to work on longer signals and divide them into subframes using winlen and winstep
        # parameters. What we do instead is, we split the sgnal ourselves into subframes, aplly the filter on the
        # subframe and put it into mfcc() function. So we get 1 mfcc row for each subframe. Because function is designed
        # to work on longer signals, it returns 2d array, each row holding 1 mfcc vector. Since we only get 1 vector,
        # our result is 2d array with the shape of (1,13). So we only get 1st row, as 1d array of 13 elements.
        mfccMatrix.append(python_speech_features.mfcc(signal=emphasized,
                                                      samplerate=fileInfo[1],
                                                      winlen=subframeDuration,
                                                      winstep=hopSize)[0])  # read above explanation for '[0]' index.
        # todo - hh: ask above design choice and/or implementation is true. we can apply the filter on the whole signal
        # todo - hh: and run mfcc() on the whole signal. Instead we do subframing first and then doing both on same time
        # print(len(mfccMatrix))
        # print(mfccMatrix)

    ''' Step II-A.4:
    For each column of the cepstrogram, DC removal is performed, resulting in the matrix M(i) i=1,2,...,N.
    '''
    for column in mfccMatrix:
        # print(np.mean(column))
        # print(column.shape)
        column = column - np.mean(column)
        # print(column)

    # print(mfccMatrix)
    allMatrixesOfExampleSet.append(mfccMatrix)

''' Step II-A.5:
A mean cepstrogram is computed by averaging the matrices of the example set, as follows:
T = 1/N * Epsilon( M(i) i=1,2,...,N )
This defines the template matrix T. In a similar manner, a variance matrix V is computed, where the distribution of
each coefficient is measured along the example set.
'''
templateMatrix = np.mean(allMatrixesOfExampleSet, axis=0)
varianceMatrix = np.var(allMatrixesOfExampleSet, axis=0)
# print(templateMatrix.shape)
# print(varianceMatrix.shape)
# print(templateMatrix)


''' Step II-A.6:
In addition to the template matrix, another feature vector is computed as follows: the matrices of the example set
are concatenated into one matrix, and the singular value decomposition (SVD) of the resulting matrix is computed.
Then, the normalized singular vector S corresponding to the largest singular value is derived. Due to the information
packing property of the SVD transform [28], the singular vector is expected to capture the most important features of
the breath event, and thus, improve the separation ability of the algorithm when used together with the template matrix
in the calculation of the breath similarity measure of test signals (see Section II-C).
'''

# todo - hh: how to choose axis here. concatanane on mfcc features or subframes?
concatanatedMatrix = np.concatenate(allMatrixesOfExampleSet, axis=0)
# todo - ai: add svg here
# https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html

print(np.shape(allMatrixesOfExampleSet))
print(concatanatedMatrix.shape)

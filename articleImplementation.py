"""
Atil Ilerialkan, 2018
Python implementation of: https://www.researchgate.net/publication/
3457766_An_Effective_Algorithm_for_Automatic_Detection_and_Exact_Demarcation_of_Breath_Sounds_in_Speech_and_Song_Signals
"""

import scipy.io.wavfile
import os
import glob
import python_speech_features
import numpy as np
from scipy import signal

# path = 'D:/atili/MMIExt/Audacity/Initial Breath Examples/'  # local copies
path = './Initial Breath Examples/'  # github copies
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


'''
II. INITIAL DETECTION ALGORITHM
'''

'''
A. Constructing the Template
'''

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
# todo - hh: how to choose axis here. concatanate on mfcc features or subframes? (concatanated on subframes for now)
# todo - hh: This decision also changes the shape of singularVector (1d array with length 13(mfcc feature count)for now)
# Concat matrix
concatanatedMatrix = np.concatenate(allMatrixesOfExampleSet, axis=0)
# print(concatanatedMatrix.shape)
# Compute SVD
singularVector = np.linalg.svd(concatanatedMatrix, compute_uv=False)
print(singularVector)
# Normalize by max norm
singularVector = singularVector / np.linalg.norm(singularVector, ord=np.inf)
# print(singularVector)
# print(np.shape(singularVector))

# print(np.shape(allMatrixesOfExampleSet))

'''
B. Detection Phase
The input for the detection algorithm is an audio signal (a monophonic recording of either speech or song, with no
background music), sampled with 44 kHz. The signal is divided into consecutive analysis frames (with a hop size of 
10 ms). For each frame, the following parameters are computed: the cepstrogram (MFCC matrix, see Fig. 4), short-time
energy, zero-crossing rate, and spectral slope (see below). Each of these is computed over a window located around the
center of the frame. A graphical plot showing the waveform of a processed signal as well as some of the parameters
computed is shown in Fig. 5.
'''

# todo - ai: read a recording here to detect.

''' Step II-B.1:
The MFCC matrix is computed as in the template generation process (see previous section). For this purpose, the
length of the MFCC analysis window used for the detection phase must match the length of the frame derived from
each breath example in the training phase.
'''
# todo - ai: implement mfcc matrix with a common function

''' Step II-B.2:
The short-time energy is computed according to the following:
E = 1/N * Epsilon(goes n=[N0, N0+(N-1)])(x^2[n])
where x[n] is the sampled audio signal, and N is the window length in samples (corresponding to 10 ms). It is
then converted to a logarithmic scale
E, dB = 10 * log10(E)
'''
# todo - ai: implement STE

''' Step II-B.3:
The zero-crossing rate (ZCR) is defined as the number of times the audio waveform changes its sign, normalized by
the window length N in samples (corresponding to 10 ms)
'''
# todo - ai: add equation for zcr here.
# todo - ai: implement ZCR

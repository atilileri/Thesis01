"""
Atil Ilerialkan, 2018
Python implementation of: https://www.researchgate.net/publication/
3457766_An_Effective_Algorithm_for_Automatic_Detection_and_Exact_Demarcation_of_Breath_Sounds_in_Speech_and_Song_Signals
"""

import numpy as np
import utils

path = utils.prefix + '/Initial Breath Examples/'

'''
II. INITIAL DETECTION ALGORITHM
'''

'''
A. Constructing the Template
'''

allMatrixesOfExampleSet = list()
exampleSet, frameLengthInSecs = utils.readExampleSet(path)
for fileInfo in exampleSet:
    # print(fileInfo)
    # print(len(fileInfo[2]))
    sampRate = fileInfo[1]
    sig = fileInfo[2]
    mfccMatrix = utils.createMfccMatrix(sig, sampRate)

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
# print(templateMatrix.shape)  # (63 subframes, 13 mfcc features)
# print(varianceMatrix.shape)  # (63 subframes, 13 mfcc features)
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
# print(singularVector)
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
center of the frame.
'''
# todo - hh: what does the last sentence above mean?

# Read input audio signal
path = utils.prefix + '/METU Recordings/hh2_48kHz_Mono_32bitFloat.wav'
fs, inputSignal = utils.readMonoWav(path)

# print('Input File:', path)
# print(fs, 'Hz, Size=', inputSignal.dtype, '*', len(inputSignal), 'bytes, Array:', inputSignal)

''' Step II-B.1:
The MFCC matrix is computed as in the template generation process (see previous section). For this purpose, the
length of the MFCC analysis window used for the detection phase must match the length of the frame(frameLengthInSecs)
derived from each breath example in the training phase.
'''
# todo - ai: change hop size to 0.010, now its 10 seconds for debugging purposes
hopSize = 10  # hop size is 10 ms for detection phase
windowLengthInSamples = 0.010 * fs  # window size is 10 ms for detection phase
for i in range(0, len(inputSignal), int(hopSize * fs)):
    # Index to stop: At the end of the analysis frame, index must stop before the file ends.
    # This assignment adjusts the last step size according to info above.
    stopIdx = min(i + int(frameLengthInSecs * fs), len(inputSignal) - 1)
    analysisFrame = inputSignal[i:stopIdx]

    mfccMatrix = utils.createMfccMatrix(analysisFrame, fs)
    # print(np.shape(mfccMatrix))
    ''' Step II-B.2:
    The short-time energy is computed according to the following:
    E = 1/N * Epsilon(goes n=[N0, N0+(N-1)])(x^2[n])
    where x[n] is the sampled audio signal, and N is the window length in samples (corresponding to 10 ms). It is
    then converted to a logarithmic scale
    E, dB = 10 * log10(E)
    '''
    # todo - ai: implement STE on center of the analysis frame

    ''' Step II-B.3:
    The zero-crossing rate (ZCR) is defined as the number of times the audio waveform changes its sign, normalized by
    the window length N in samples (corresponding to 10 ms)
    '''
    # todo - ai: add equation for zcr here.
    # todo - ai: implement ZCR on center of the analysis frame

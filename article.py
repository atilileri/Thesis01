"""
Atil Ilerialkan, 2018
Python implementation of: https://www.researchgate.net/publication/
3457766_An_Effective_Algorithm_for_Automatic_Detection_and_Exact_Demarcation_of_Breath_Sounds_in_Speech_and_Song_Signals
"""

import numpy as np
import utils

path = utils.prefix + '/Initial Breath Examples/'
# path = utils.prefix + '/METU Recordings/hh2_breath/'
# path = utils.prefix + '/METU Recordings/'

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

''' Step II-A.6:
In addition to the template matrix, another feature vector is computed as follows: the matrices of the example set
are concatenated into one matrix, and the singular value decomposition (SVD) of the resulting matrix is computed.
Then, the normalized singular vector S corresponding to the largest singular value is derived. Due to the information
packing property of the SVD transform [28], the singular vector is expected to capture the most important features of
the breath event, and thus, improve the separation ability of the algorithm when used together with the template matrix
in the calculation of the breath similarity measure of test signals (see Section II-C).
'''
# Concat matrix
concatanatedMatrix = np.concatenate(allMatrixesOfExampleSet, axis=0)
# Compute SVD
singularVectors, singularValues, _ = np.linalg.svd(concatanatedMatrix.transpose(), full_matrices=True)
mainSingularVector = singularVectors[np.argmax(np.abs(singularValues))]
# print('allMatrixesOfExampleSet:', np.shape(allMatrixesOfExampleSet))  # (24 files, 63 subframes, 13 mfcc features)
# print('templateMatrix:', templateMatrix.shape)  # (63 subframes, 13 mfcc features)
# print('varianceMatrix:', varianceMatrix.shape)  # (63 subframes, 13 mfcc features)
# print('concatanatedMatrix:', concatanatedMatrix.shape)
# print('singularVector:', np.shape(mainSingularVector))

# get normalized Singular Vector for calculations of Cn
normSingVect = mainSingularVector / max(np.abs(mainSingularVector))

'''
B. Detection Phase
The input for the detection algorithm is an audio signal (a monophonic recording of either speech or song, with no
background music), sampled with 44 kHz. The signal is divided into consecutive analysis frames (with a hop size of 
10 ms). For each frame, the following parameters are computed: the cepstrogram (MFCC matrix, see Fig. 4), short-time
energy, zero-crossing rate, and spectral slope (see below). Each of these is computed over a window located around the
center of the frame.
'''
# Read input audio signal
# Note that this file can not be added to github, because it is larger than the size limit(100MB)
path = './whole_speech.wav'
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
windowLengthInSamples = int(0.010 * fs)  # window size is 10 ms for detection phase
for i in range(0, len(inputSignal), int(hopSize * fs)):
    # Index to stop: At the end of the analysis frame, index must stop before the file ends.
    # This assignment adjusts the last step size according to info above.
    # print(len(inputSignal))
    # print(i + int(frameLengthInSecs * fs))
    stopIdx = min(i + int(frameLengthInSecs * fs), len(inputSignal) - 1)

    analysisFrame = inputSignal[i:stopIdx]
    centerWindow = utils.getCenterWindow(analysisFrame, windowLengthInSamples)

    # todo - ai : createMfccMatrix() should work with centerWindow, not analysisFrame. Throws error now. Investigate!
    # The Cepstrogram (MFCC matrix) is computed over a window located around the center of the frame
    cepstogramXi = utils.createMfccMatrix(analysisFrame, fs)
    # print(np.shape(mfccMatrix))
    ''' Step II-B.2:
    The short-time energy is computed according to the following:
    E = 1/N * Epsilon(goes n=[N0, N0+(N-1)])(x^2[n])
    where x[n] is the sampled audio signal, and N is the window length in samples (corresponding to 10 ms). It is
    then converted to a logarithmic scale
    E, dB = 10 * log10(E)
    '''
    # Short Time Energy is computed over a window located around the center of the frame
    steXi, db = utils.calcShortTimeEnergy(centerWindow)
    ''' Step II-B.3:
    The zero-crossing rate (ZCR) is defined as the number of times the audio waveform changes its sign, normalized by
    the window length N in samples (corresponding to 10 ms)
    ZCR = 1/N * Epsilon(goes n=[N0+1, N0+(N-1)])( 0.5 * abs( sign(x[n]) - sign(x[n-1]) ) )
    '''
    # Zero Crossing Rate is computed over a window located around the center of the frame
    zcrXi = utils.calcZeroCrossingRate(centerWindow)
    ''' Step II-B.4:
    The spectral slope is computed by taking the discrete Fourier transform of the analysis window, evaluating its
    magnitude at frequencies of pi/2 and pi (corresponding here to 11 and 22 kHz, respectively), and computing the
    slope of the straight line fit between these two points. It is known that in voiced speech most of the spectral
    energy is contained in the lower frequencies (below 4 kHz). Therefore, in voiced speech, the spectrum is expected to 
    be rather flat between 11 and 22 kHz. In periods of silence, the waveform is close to random, which also leads to a
    relatively flat spectrum throughout the entire band. This suggests that the spectral slope in voiced/silence parts
    would yield low values, when measured as described previously. On the other hand, in breath sounds, like in most 
    unvoiced phonemes, there is still a significant amount of energy in the middle frequency band (10–15 kHz) and
    relatively low energy in the high band (22 kHz). Thus, the spectral slope is expected to be steeper, and could be
    used to differentiate between voiced/silence and unvoiced/breath. As such, the spectral slope is used here as an
    additional parameter for identifying the edges of the breath (see Section III).
    '''
    slopeXi = utils.calcSpectralSlope(centerWindow, fs)

    '''
    C.Computation of the Breath Similarity Measure
    Once the aforementioned parameters are computed for a given frame Xi, its short-time cepstrogram (MFCC matrix) is
    used for calculating its breath similarity measure. The similarity measure, denoted B(Xi, T, V, S), is computed between 
    the cepstrogram of the frame, M(Xi), the template cepstrogram T (with V being the variance matrix) and the singular 
    vector S . The steps of the computation are as follows (Fig. 6):
    '''
    ''' Step II-C.1:
    The normalized difference matrix  D = (M(Xi) - T) / V  is computed. The normalization (element-by-element) by the 
    variance matrix is performed in order to compensate for the differences in the distributions of the various cepstral
    coefficients.
    '''
    diffXi = np.subtract(cepstogramXi, templateMatrix)
    normDiffXi = np.divide(diffXi, varianceMatrix)

    # Reminder Here: The cepstrogram is defined as a matrix whose columns are the MFCC vectors for each subframe.
    ''' Step II-C.2:
    The difference matrix is liftered by multiplying each column with a half-Hamming window that emphasizes the lower
    cepstral coefficients. It has been found in preliminary experiments, that this procedure yields better separation
    between breath sounds and other sounds (see also [2]).
    '''
    lifter = np.hamming(26)[13:]  # get half-Hamming window
    liftNormDiffXi = np.multiply(normDiffXi, lifter)

    ''' Step II-C.3:
    A first similarity measure(Cp) is computed by taking the inverse of the sum of squares of all elements of the
    normalized difference matrix, according to the following equation:
        Cp = 1 / Epsilon(goes i=[1,n], j=[1, Ne])(Dij^2)
    where n is the number of subframes, and Ne is the number of MFC coefficients computed for each subframe.
    When the cepstrogram is very similar to the template, the elements of the difference matrix should be small, leading
    to a high value of this similarity measure. When the frame contains a signal which is very different from breath,
    the measure is expected to yield small values. This template matching procedure with a scaled Euclidean distance is
    essentially a special case of a two-class Gaussian classifier [30] with a diagonal covariance matrix. This is due to
    the computation of the MFCC, which involves a discrete cosine transform as its last step [20], known for its
    tendency to decorrelate the mel-scale filter log-energies [22].
    '''
    squaresXi = np.square(liftNormDiffXi)
    sumOfSquaresXi = np.sum(squaresXi)
    cp = 1 / sumOfSquaresXi

    ''' Step II-C.4:
    A second similarity measure(Cn) is computed by taking the sum of the inner products between the singular vector
    (see Section II-A) and the normalized columns of the cepstrogram. Since the singular vector is assumed to capture
    the important characteristics of breath sounds, these inner products (and, therefore, Cn) are expected to be small
    when the frame contains information from other phonemes.
    '''
    innerProducts = []
    for col in cepstogramXi:
        normCol = col / np.max(np.abs(col))
        innerProduct = np.inner(normCol, normSingVect)
        innerProducts.append(innerProduct)

    cn = np.sum(innerProducts)

    print('Similarity Measure for frame', i, ':', cp * cn)

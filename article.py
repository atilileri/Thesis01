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
Section II. INITIAL DETECTION ALGORITHM
'''

'''
Section II-A : Constructing the Template
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

''' Section II-A.5:
A mean cepstrogram is computed by averaging the matrices of the example set, as follows:
T = 1/N * Epsilon( M(i) i=1,2,...,N )
This defines the template matrix T. In a similar manner, a variance matrix V is computed, where the distribution of
each coefficient is measured along the example set.
'''
templateMatrix = np.mean(allMatrixesOfExampleSet, axis=0)
varianceMatrix = np.var(allMatrixesOfExampleSet, axis=0)

''' Section II-A.6:
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

print('Template & Variance Matrix, Singular Vector are Constructed')

''' Section II-B : Detection Phase
The input for the detection algorithm is an audio signal (a monophonic recording of either speech or song, with no
background music), sampled with 44 kHz. The signal is divided into consecutive analysis frames (with a hop size of 
10 ms). For each frame, the following parameters are computed: the cepstrogram (MFCC matrix, see Fig. 4), short-time
energy, zero-crossing rate, and spectral slope (see below). Each of these is computed over a window located around the
center of the frame.
'''

# Go over example set again and find params for Breath Similarity Measurements in "Step II-C.Detection"
bsmArray = []
for cepsEach in allMatrixesOfExampleSet:
    bsmEach = utils.calcBSM(cepsEach, templateMatrix, varianceMatrix, normSingVect)
    bsmArray.append(bsmEach)
bm = min(bsmArray)
bsmThreshold = bm / 2

# Read input audio signal
# Note that this file can not be added to github, because it is larger than the size limit(100MB)
path = './whole_speech.wav'
fs, inputSignal = utils.readMonoWav(path)

# print('Input File:', path)
# print(fs, 'Hz, Size=', inputSignal.dtype, '*', len(inputSignal), 'bytes, Array:', inputSignal)

# todo - ai: change hop size to 0.010, now its 10 seconds for debugging purposes
hopSize = 10  # hop size is 10 ms for detection phase
for i in range(0, len(inputSignal), int(hopSize * fs)):
    # MyNote 1 - ai : Since there is no explicit information on the length of each consecutive analysis frame in
    #  detection phase, a minimum value is taken as analysis frame size. Which is the length of the frame
    #  (frameLengthInSecs) derived from each breath example in the training phase.
    # todo - hh : ask if frameLengthInSecs | there is a general frame length in the literature?
    # Index to stop: At the end of the analysis frame, index must stop before the file ends.
    # This assignment adjusts the last step size according to info above.
    stopIdx = min(i + int(frameLengthInSecs * fs), len(inputSignal) - 1)
    analysisFrame = inputSignal[i:stopIdx]

    ''' Section II-B.1:
    The MFCC matrix is computed as in the template generation process (see previous section). For this purpose, the
    length of the MFCC analysis window used for the detection phase must match the length of the frame 
    (frameLengthInSecs) derived from each breath example in the training phase.
    '''
    # The Cepstrogram (MFCC matrix) is computed over a window located around the center of the frame
    windowLengthInSamples = len(analysisFrame)  # window size is frameLengthInSecs for mfcc calculations
    centerWindow = utils.getCenterWindow(analysisFrame, windowLengthInSamples)
    # Since we are using frameLengthInSecs as the analysis frame length(Refer to "Note 1" above), we do not need to
    # get the center window of the frame with its own length :) but, we are doing it for generality of the code.
    # This decision makes sense when frame length in Detection Phase is changed to a value other than MFCC analysis
    # frame length. Note 1 states this also. But for now, centerWindow is exactly same as analysisFrame in below :)
    cepstogramXi = utils.createMfccMatrix(centerWindow, fs)
    ''' Section II-B.2:
    The short-time energy is computed according to the following:
    E = 1/N * Epsilon(goes n=[N0, N0+(N-1)])(x^2[n])
    where x[n] is the sampled audio signal, and N is the window length in samples (corresponding to 10 ms). It is
    then converted to a logarithmic scale
    E, dB = 10 * log10(E)
    '''
    # Short Time Energy is computed over a window located around the center of the frame
    windowLengthInSamples = int(0.010 * fs)  # window size is 10 ms for STE calculations
    centerWindow = utils.getCenterWindow(analysisFrame, windowLengthInSamples)
    steXi, db = utils.calcShortTimeEnergy(centerWindow)
    ''' Section II-B.3:
    The zero-crossing rate (ZCR) is defined as the number of times the audio waveform changes its sign, normalized
    by the window length N in samples (corresponding to 10 ms)
    ZCR = 1/N * Epsilon(goes n=[N0+1, N0+(N-1)])( 0.5 * abs( sign(x[n]) - sign(x[n-1]) ) )
    '''
    # Zero Crossing Rate is computed over a window located around the center of the frame
    windowLengthInSamples = int(0.010 * fs)  # window size is 10 ms for ZCR calculations
    centerWindow = utils.getCenterWindow(analysisFrame, windowLengthInSamples)
    zcrXi = utils.calcZeroCrossingRate(centerWindow)
    ''' Section II-B.4:
    The spectral slope is computed by taking the discrete Fourier transform of the analysis window, evaluating its
    magnitude at frequencies of pi/2 and pi (corresponding here to 11 and 22 kHz, respectively), and computing the
    slope of the straight line fit between these two points. It is known that in voiced speech most of the spectral
    energy is contained in the lower frequencies (below 4 kHz). Therefore, in voiced speech, the spectrum is
    expected to be rather flat between 11 and 22 kHz. In periods of silence, the waveform is close to random, which
    also leads to a relatively flat spectrum throughout the entire band. This suggests that the spectral slope in
    voiced/silence parts would yield low values, when measured as described previously. On the other hand, in breath
    sounds, like in most unvoiced phonemes, there is still a significant amount of energy in the middle frequency 
    band (10–15 kHz) and relatively low energy in the high band (22 kHz). Thus, the spectral slope is expected to be
    steeper, and could be used to differentiate between voiced/silence and unvoiced/breath. As such, the spectral
    slope is used here as an additional parameter for identifying the edges of the breath (see Section III).
    '''
    # The Spectral Slope is computed over a window located around the center of the frame
    # MyNote 2 - ai : window length is not referred exactly in the article so, whole frame is taken, same as mfcc.
    # todo - hh : ask if 10 ms | mfcc analysis window length | full analysis frame ? (last 2 are same for now)
    windowLengthInSamples = len(analysisFrame)  # window size is frameLengthInSecs for slope calculations
    centerWindow = utils.getCenterWindow(analysisFrame, windowLengthInSamples)
    slopeXi = utils.calcSpectralSlope(centerWindow, fs)
    
    ''' Section II-C : Computation of the Breath Similarity Measure
    Once the aforementioned parameters are computed for a given frame Xi, its short-time cepstrogram (MFCC matrix)
    is used for calculating its breath similarity measure. The similarity measure, denoted B(Xi, T, V, S), is 
    computed between the cepstrogram of the frame, M(Xi), the template cepstrogram T (with V being the variance
    matrix) and the singular vector S . The steps of the computation are as follows (Fig. 6):
    '''
    bsmXi = utils.calcBSM(cepstogramXi, templateMatrix, varianceMatrix, normSingVect)

    ''' Section II-C.Detection
    The breath detection involves a two-step decision. The initial decision treats each frame independently of other
    frames and classifies each as breathy/not breathy based on its similarity measure B(Xi, T, V, S), energy and
    zero-crossing rate. A frame is initially classified as breathy if all three of the following occur:
    1) The breath similarity measure is above a given threshold. This threshold is initially set in the learning phase,
    during the template construction, when the breath similarity measure B(Xi, T, V, S) is computed for each of the
    examples. The minimum value of the similarity measures between each of the examples and the template is determined, 
    denoted by Bm. The threshold is set to Bm / 2. The logic behind this setting is that the frame-to-template
    similarity of breath sounds in general is expected to be somewhat lower than the similarity among examples used to
    construct the template in the first place. This parameter is referred as "bsmThreshold".
    2) The energy is below a given threshold, which is chosen to be below the average energy of voiced speech (see
    Section III-A). This parameter is referred as "engThreshold".
    3) The zero-crossing rate is below a given threshold. Experimental data have shown that ZCR above 0.25 (assuming
    a sampling rate of 44 kHz) is exhibited only by a number of unvoiced fricatives, and breath sounds have much lower
    ZCR (see Section III-A). This parameter is referred as "zcrThreshold".
    '''
    zcrThreshold = 0.25
    engThreshold = 999999  # todo - ai : calculate this. Pseudo for now
    if bsmXi > bsmThreshold and zcrXi < zcrThreshold and steXi < engThreshold:
        print(i, '-', stopIdx, ': BREATH')
    else:
        print(i, '-', stopIdx, ': no')

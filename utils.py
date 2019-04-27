"""
utils for common use
"""
import scipy.io.wavfile
import numpy as np
from scipy import signal
import python_speech_features
import os
import glob
import pickle

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
    # Convert multi channel record to mono
    if len(samples.shape) > 1 and 1 < samples.shape[1]:  # python evals expressions LTR so it's safe to write this
        # Solution 1: Convert to mono by taking mean of the channels
        # samplesMono = np.mean(samples, axis=1, dtype=samples.dtype)
        # Solution 2: Convert to mono by taking only first channel
        samplesMono = samples[:, 0]
        # todo - ai : all channels can be used as different inputs to augment the data? For now, get first channel only.
    else:
        samplesMono = samples
    # print('File: \'', pathToFile, '\', Length: ', len(samplesMono)/samplingRate, 'seconds.')
    return samplingRate, samplesMono


# reads wav files in the given folder
def readExampleSet(folderPath):
    exampleSet = []
    minlen = 999999  # pseudo big number
    sampleRate = 0
    '''
    Section II-A.1:
    Several signals containing isolated breath examples are selected, forming the example set. From each example, a 
    section of fixed length, typically equal to the length of the shortest example in the set (about 100–160 ms), is
    derived. This length is used throughout the algorithm as the frame length (see Section II-C).
    '''
    for filePath in glob.glob(os.path.join(folderPath, '*.wav')):
        sampleRate, samples = readMonoWav(filePath)

        exampleSet.append([filePath, sampleRate, samples])
        if len(samples) < minlen:
            minlen = len(samples)
            # todo - ai: minlen diff looks huge, create better samples
            # print('min frame:', minlen)

    # fix frame length. Refer to Section B.1
    for file in exampleSet:
        file[2] = getCenterWindow(file[2], minlen)

    return exampleSet, minlen/sampleRate


# function to create mfcc  matrix of a audio file
def createMfccMatrix(sig,
                     sampRate,
                     subframeDuration=0.010,  # 10 ms
                     hopSize=0.005):  # 5 ms
    subframeDurationBySample = int(subframeDuration * sampRate)
    alpha = 0.95

    ''' Section II-A.2:
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
        emphasized = signal.lfilter([1, -1 * alpha], [1., 1.], sig[i:stopIdx])

        ''' Section II-A.3:
        For each breath example, the MFCC are computed for every subframe, thus forming a short-time cepstrogram
        representation of the example. The cepstrogram is defined as a matrix whose columns are the MFCC vectors
        for each subframe. Each such matrix is denoted by M(i), i=1,2,...,N where N is the number of examples in the
        examples set. The construction of the cepstrogram is demonstrated in Fig. 4.
        '''
        # http://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.mfcc
        # mfcc() function is designed to work on longer signals and divide them into subframes using winlen and winstep
        # parameters. What we do instead is, we split the signal ourselves into subframes, apply the filter on the
        # subframe and put it into mfcc() function. So we get 1 mfcc row for each subframe. Because function is designed
        # to work on longer signals, it returns 2d array, each row holding 1 mfcc vector. Since we only get 1 vector,
        # our result is 2d array with the shape of (1,13). So we only get 1st row, as 1d array of 13 elements.
        mfccMatrix.append(python_speech_features.mfcc(signal=emphasized,
                                                      samplerate=sampRate,
                                                      winlen=subframeDuration,
                                                      winstep=hopSize)[0])  # read above explanation for '[0]' index.
        # print(len(mfccMatrix))
        # print(mfccMatrix)

    ''' Section II-A.4:
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


def getCenterWindow(frame, windowLengthInSamples):
    midIndex = len(frame) / 2
    halfWindowLen = min(midIndex, windowLengthInSamples / 2)
    # we cast to int below, instead of above(where halfWindowLen is defined)
    # because otherwise our slice is one element short, for odd windowLength values
    centerWindow = frame[int(midIndex-halfWindowLen):int(midIndex+halfWindowLen)]

    return centerWindow


def calcShortTimeEnergy(window):
    """ calculates short-time energy of a given window
    The short-time energy is computed according to the following:
    E = 1/N * Epsilon(goes n=[N0, N0+(N-1)])(x^2[n])
    where x[n] is the sampled audio signal, and N is the window length in samples (corresponding to 10 ms). It is
    then converted to a logarithmic scale
    E, dB = 10 * log10(E)
    :param window: given window
    :return: short time energy of the window
    """
    squared = np.square(window, dtype=type(window[0]))
    summed = np.sum(squared, dtype=type(window[0]))
    shortTimeEnergy = summed / len(window)
    # Have it also in dB
    energyInDb = 10 * np.log10(shortTimeEnergy + 1e-8)
    return shortTimeEnergy, energyInDb


def calcZeroCrossingRate(window):
    """ calculates zero-crossing rate of a given window
     The zero-crossing rate (ZCR) is defined as the number of times the audio waveform changes its sign, normalized by
    the window length N in samples (corresponding to 10 ms)
    ZCR = 1/N * Epsilon(goes n=[N0+1, N0+(N-1)])( 0.5 * abs( sign(x[n]) - sign(x[n-1]) ) )
    :param window: given window
    :return: zero crossing rate of the window
    """
    zeroCrossings = 0
    for i in range(1, len(window)):
        zeroCrossings += 0.5 * np.abs(np.sign(window[i]) - np.sign(window[i-1]))
    return zeroCrossings / len(window)


def calcSpectralSlope(window, fs):
    """ calculates spectral slope of a given window
    The spectral slope is computed by taking the discrete Fourier transform of the analysis window, evaluating its
    magnitude at frequencies of pi/2 and pi (corresponding here to 11 and 22 kHz, respectively), and computing the
    slope of the straight line fit between these two points.
    :param window: given window
    :param fs: sampling rate
    :return: spectral slope of the window
    """
    # todo - hh: ask if implementation is right
    magnitudes = np.abs(np.fft.fft(window))
    frequencies = np.fft.fftfreq(len(window), 1/fs)
    # find closest frequency index to fs/4
    indexPiOver2 = np.argmin(np.abs(frequencies - fs // 4))  # index at fs/4 (corresponding to pi/2)
    indexPi = np.argmin(np.abs(frequencies - fs // 2))  # index at fs/2 (corresponding to pi)
    y1 = magnitudes[indexPiOver2]
    y2 = magnitudes[indexPi]
    x1 = frequencies[indexPiOver2]
    x2 = frequencies[indexPi]
    # todo - ai: I added 10k factor below to make number compareable. Check for future
    return np.abs((y2-y1) / ((x2-x1)/10000))


def calcBSM(cepstogram, templateMatrix, varianceMatrix, normSingVect):
    """ This function calculates Breath Similarity Measure for given frame
    :param cepstogram: cepstogram of the frame (also reffered as M)
    :param templateMatrix: the template cepstogram (also reffered as T)
    :param varianceMatrix: the variance matrix (also reffered as V)
    :param normSingVect: the singular vector, normalized (also reffered as S)
    :return: bsm - Breath Similarity Measure of the frame
    """

    ''' Section II-C.1:
    The normalized difference matrix  D = (M(Xi) - T) / V  is computed. The normalization (element-by-element) by the 
    variance matrix is performed in order to compensate for the differences in the distributions of the various cepstral
    coefficients.
    '''
    diffXi = np.subtract(cepstogram, templateMatrix)
    normDiffXi = np.divide(diffXi, varianceMatrix)

    # Reminder Here: The cepstrogram is defined as a matrix whose columns are the MFCC vectors for each subframe.
    ''' Section II-C.2:
    The difference matrix is liftered by multiplying each column with a half-Hamming window that emphasizes the lower
    cepstral coefficients. It has been found in preliminary experiments, that this procedure yields better separation
    between breath sounds and other sounds (see also [2]).
    '''
    lifter = np.hamming(26)[13:]  # get half-Hamming window
    liftNormDiffXi = np.multiply(normDiffXi, lifter)

    ''' Section II-C.3:
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

    ''' Section II-C.4:
    A second similarity measure(Cn) is computed by taking the sum of the inner products between the singular vector
    (see Section II-A) and the normalized columns of the cepstrogram. Since the singular vector is assumed to capture
    the important characteristics of breath sounds, these inner products (and, therefore, Cn) are expected to be small
    when the frame contains information from other phonemes.
    '''
    # todo - hh : there are negative values in both cn and singVect. How to implement?
    innerProducts = []
    for col in cepstogram:
        # column normalization
        normCol = col / np.sqrt(np.sum(np.square(col)))
        innerProduct = np.inner(normCol, normSingVect)
        innerProducts.append(innerProduct)

    cn = np.sum(innerProducts)

    ''' Section II-C.5:
    The final breath similarity measure is defined as the product of the two measures: . It was found experimentally
    that this combination of similarity measures yields better separation between breath and nonbreath than using
    just the difference matrix or the singular vector.
    '''
    # todo - hh : there are negative values in Cn. But in initial classification section, there is a threshold value
    #  to pass for detection. How negative values can pass this threshold. There is a 'reciprocal of breath similarity
    #  function' statement in figure 5
    bsm = cp * cn
    # print('Similarity Measure for frame:', bsm, '=', cp, '*', cn)
    return bsm


# Calculates p-point running average(moving mean) on inputArray
def calcRunningAvg(inputArray, p):
    runAvg = []
    if len(inputArray) < p:  # only return average, if input array is smaller than point-size
        runAvg.append(np.average(inputArray))
    else:
        for i in range(p, len(inputArray)+1):  # +1 is because slicing does not include last member in the line below.
            runAvg.append(np.average(inputArray[i - p:i]))
    return runAvg


# Saves variable data to file
def saveData(data, path):
    pickle.dump(data, open(path, "wb"))


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, "rb"))


def calcTemplateParameters(exampleSetPath, resultSavePath):
    templateParameters = {}

    allMatrixesOfExampleSet = list()
    exampleSet, frameLengthInSecs = readExampleSet(exampleSetPath)
    templateParameters['frameLengthInSecs'] = frameLengthInSecs
    for fileInfo in exampleSet:
        # print(fileInfo)
        # print(len(fileInfo[2]))
        sampRate = fileInfo[1]
        sig = fileInfo[2]
        mfccMatrix = createMfccMatrix(sig, sampRate)

        # print(mfccMatrix)
        allMatrixesOfExampleSet.append(mfccMatrix)

    templateParameters['cepstograms'] = allMatrixesOfExampleSet
    ''' Section II-A.5:
    A mean cepstrogram is computed by averaging the matrices of the example set, as follows:
    T = 1/N * Epsilon( M(i) i=1,2,...,N )
    This defines the template matrix T. In a similar manner, a variance matrix V is computed, where the distribution of
    each coefficient is measured along the example set.
    '''
    templateMatrix = np.mean(allMatrixesOfExampleSet, axis=0)
    templateParameters['templateMatrix'] = templateMatrix

    varianceMatrix = np.var(allMatrixesOfExampleSet, axis=0)
    templateParameters['varianceMatrix'] = varianceMatrix
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
    normSingVect = mainSingularVector / np.sqrt(np.sum(np.square(mainSingularVector)))
    templateParameters['normalizedSingularVector'] = normSingVect

    # Go over example set again and find threshold for Breath Similarity Measurements in "Step II-C.Detection"
    '''
    This threshold is initially set in the learning phase, during the template construction, when the breath similarity
    measure B(Xi, T, V , S) is computed for each of the examples. The minimum value of the similarity measures between each
    of the examples and the template is determined, denoted by Bm. The threshold is set to Bm / 2.
    '''
    bsmArray = []
    for cepsEach in allMatrixesOfExampleSet:
        bsmEach = calcBSM(cepsEach, templateMatrix, varianceMatrix, normSingVect)
        bsmArray.append(bsmEach)
    bm = min(bsmArray)
    bsmThreshold = bm / 2
    templateParameters['bsmThreshold'] = bsmThreshold

    saveData(templateParameters, resultSavePath)


def calcInitialClassifications(inputFilePath, templateMatrix, varianceMatrix,
                               normSingVect, frameLengthInSecs, bsmThreshold, resultSavePath):
    # Read input audio signal
    fs, inputSignal = readMonoWav(inputFilePath)

    # print('Input File:', path)
    # print(fs, 'Hz, Size=', inputSignal.dtype, '*', len(inputSignal), 'bytes, Array:', inputSignal)

    # todo - ai: change hop size to 0.010, now its 1 seconds for debugging purposes
    hopSize = 1  # hop size is 10 ms for detection phase
    initialClassifications = []
    frameParameters = {}
    for i in range(0, len(inputSignal), int(hopSize * fs)):
        # MyNote 1 - ai : Since there is no explicit information on the length of each consecutive analysis frame in
        #  detection phase, a minimum value is taken as analysis frame size. Which is the length of the frame
        #  (frameLengthInSecs) derived from each breath example in the training phase.
        # todo - hh : ask if frameLengthInSecs | there is a general frame length in the literature?
        stopIdx = i + int(frameLengthInSecs * fs)
        print('\rCalculating parameters for frame', i, '-', stopIdx, 'of', len(inputSignal), '...', end=' ')
        analysisFrame = inputSignal[i:stopIdx]
        frameParameters['sampleRate'] = fs
        frameParameters['startSampleIndex'] = i
        frameParameters['endSampleIndex'] = stopIdx
        ''' Section II-B.1:
        The MFCC matrix is computed as in the template generation process (see previous section). For this purpose, the
        length of the MFCC analysis window used for the detection phase must match the length of the frame 
        (frameLengthInSecs) derived from each breath example in the training phase.
        '''
        # The Cepstrogram (MFCC matrix) is computed over a window located around the center of the frame
        windowLengthInSamples = len(analysisFrame)  # window size is frameLengthInSecs for mfcc calculations
        centerWindow = getCenterWindow(analysisFrame, windowLengthInSamples)
        # Since we are using frameLengthInSecs as the analysis frame length(Refer to "Note 1" above), we do not need to
        # get the center window of the frame with its own length :) but, we are doing it for generality of the code.
        # This decision makes sense when frame length in Detection Phase is changed to a value other than MFCC analysis
        # frame length. Note 1 states this also. But for now, centerWindow is exactly same as analysisFrame in below :)
        cepstogramXi = createMfccMatrix(centerWindow, fs)
        ''' Section II-B.2:
        The short-time energy is computed according to the following:
        E = 1/N * Epsilon(goes n=[N0, N0+(N-1)])(x^2[n])
        where x[n] is the sampled audio signal, and N is the window length in samples (corresponding to 10 ms). It is
        then converted to a logarithmic scale
        E, dB = 10 * log10(E)
        '''
        # Short Time Energy is computed over a window located around the center of the frame
        windowLengthInSamples = int(0.010 * fs)  # window size is 10 ms for STE calculations
        centerWindow = getCenterWindow(analysisFrame, windowLengthInSamples)
        steXi, steInDbXi = calcShortTimeEnergy(centerWindow)
        frameParameters['shortTimeEnergy'] = steInDbXi
        ''' Section II-B.3:
        The zero-crossing rate (ZCR) is defined as the number of times the audio waveform changes its sign, normalized
        by the window length N in samples (corresponding to 10 ms)
        ZCR = 1/N * Epsilon(goes n=[N0+1, N0+(N-1)])( 0.5 * abs( sign(x[n]) - sign(x[n-1]) ) )
        '''
        # Zero Crossing Rate is computed over a window located around the center of the frame
        windowLengthInSamples = int(0.010 * fs)  # window size is 10 ms for ZCR calculations
        centerWindow = getCenterWindow(analysisFrame, windowLengthInSamples)
        zcrXi = calcZeroCrossingRate(centerWindow)
        frameParameters['zeroCrossingRate'] = zcrXi
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
        centerWindow = getCenterWindow(analysisFrame, windowLengthInSamples)
        slopeXi = calcSpectralSlope(centerWindow, fs)
        frameParameters['slope'] = slopeXi
        ''' Section II-C : Computation of the Breath Similarity Measure
        Once the aforementioned parameters are computed for a given frame Xi, its short-time cepstrogram (MFCC matrix)
        is used for calculating its breath similarity measure. The similarity measure, denoted B(Xi, T, V, S), is 
        computed between the cepstrogram of the frame, M(Xi), the template cepstrogram T (with V being the variance
        matrix) and the singular vector S . The steps of the computation are as follows (Fig. 6):
        '''
        bsmXi = calcBSM(cepstogramXi, templateMatrix, varianceMatrix, normSingVect)
        frameParameters['breathSimilarityMeasure'] = bsmXi
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
        Following the initial detection, a binary breathiness index is assigned to each frame: breathy frames are assigned
        index 1, whereas nonbreathy frames are assigned index 0.
        '''
        zcrThreshold = 0.25
        # todo - hh : ask what can we assign to energy threshold, according to fig. 7
        engThreshold = 999999  # todo - ai : calculate this. Pseudo for now
        if bsmXi > bsmThreshold and zcrXi < zcrThreshold and steXi < engThreshold:
            # print(i, '-', stopIdx, ': BREATH')
            frameParameters['breathinessIndice'] = 1
        else:
            # print(i, '-', stopIdx, ': no')
            frameParameters['breathinessIndice'] = 0
        initialClassifications.append(frameParameters.copy())  # copy before clear the temp parameter dictionary
        frameParameters.clear()
    print('\nInitial Classifications are Done.')

    saveData(initialClassifications, resultSavePath)

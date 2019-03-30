"""
Atil Ilerialkan, 2018
Python implementation of: https://www.researchgate.net/publication/
3457766_An_Effective_Algorithm_for_Automatic_Detection_and_Exact_Demarcation_of_Breath_Sounds_in_Speech_and_Song_Signals
"""

import numpy as np
import utils
import scipy.signal as sp
import matplotlib.pyplot as plt

rebuildTemplateParams = False
rebuildInitialClassifications = False
templateParamsPath = './CodeDataStorage/templateParameters.pickle'
initialClassificationsPath = './CodeDataStorage/initialClassifications.pickle'
'''
Section II. INITIAL DETECTION ALGORITHM
'''

'''
Section II-A : Constructing the Template
'''
if rebuildTemplateParams:
    path = utils.prefix + '/Initial Breath Examples/'
    # path = utils.prefix + '/METU Recordings/hh2_breath/'
    # path = utils.prefix + '/METU Recordings/'
    utils.calcTemplateParameters(exampleSetPath=path, resultSavePath=templateParamsPath)

templateParameters = utils.loadData(templateParamsPath)
allMatrixesOfExampleSet = templateParameters['cepstograms']
templateMatrix = templateParameters['templateMatrix']
varianceMatrix = templateParameters['varianceMatrix']
normSingVect = templateParameters['normalizedSingularVector']
frameLengthInSecs = templateParameters['frameLengthInSecs']
bsmThreshold = templateParameters['bsmThreshold']
print('Template Parameters are Constructed')

''' Section II-B : Detection Phase
The input for the detection algorithm is an audio signal (a monophonic recording of either speech or song, with no
background music), sampled with 44 kHz. The signal is divided into consecutive analysis frames (with a hop size of 
10 ms). For each frame, the following parameters are computed: the cepstrogram (MFCC matrix, see Fig. 4), short-time
energy, zero-crossing rate, and spectral slope (see below). Each of these is computed over a window located around the
center of the frame.
'''
if rebuildInitialClassifications:
    # Note that this file can not be added to github, because it is larger than the size limit(100MB)
    path = './whole_speech.wav'
    # todo - ai: fix performance
    utils.calcInitialClassifications(path, templateMatrix, varianceMatrix,
                                     normSingVect, frameLengthInSecs, bsmThreshold,
                                     resultSavePath=initialClassificationsPath)

initialClassifications = utils.loadData(initialClassificationsPath)
print('Initial Classifications are Done.')

# print initial classifications
for frame in initialClassifications:
    fs = frame['sampleRate']
    startIdx = frame['startSampleIndex']
    endIdx = frame['endSampleIndex']
    print(startIdx, '(', startIdx/fs, ') -', end=' ')
    print(endIdx, '(', endIdx/fs, '):', end=' ')
    if frame['breathinessIndice'] == 1:
        print('BREATH')
    elif frame['breathinessIndice'] == 0:
        print('no')

'''
Section III. EDGE DETECTION AND FALSE ALARM ELIMINATION
'''

''' Section III-A : General Approach to False Detection Elimination
In both of the edge detection algorithms presented here, similar criteria are used for rejection of false positives.
1) Preliminary Duration Threshold: A breath event is expected to yield a significant peak in the contour of the breath 
similarity measure function, i.e., a considerable number of frames that rise above the similarity threshold. If the 
number of such frames is too low, it is likely to be a false detection.
2) Upper Energy Threshold: Typically, the local energy within a breath epoch is much lower than that of voiced speech 
and somewhat lower than most of the unvoiced speech (see Fig. 7). Hence, if some frames in the detected epoch, after 
edge marking, exceed a predefined energy threshold, the section should be rejected.
3) Lower ZCR Threshold: Since most of the breaths are unvoiced sounds, the ZCR during a breath event is expected to be 
in the same range as that of most unvoiced phonemes, i.e., higher than that of voiced phonemes. This was empirically 
verified. Therefore, if the ZCR throughout the entire marked section is beneath a ZCR lower threshold, it will be 
rejected as a probable voiced phoneme.
4) Upper ZCR Threshold: It is known that certain unvoiced phonemes, such as fricative consonants, can exhibit a high ZCR
(0.3–0.4 given a 44-kHz sampling frequency). Preliminary experiments showed that the maximum ZCR of breath sounds is
considerably lower (see Fig. 8). An upper threshold on the ZCR can thus prevent false detections of certain fricatives 
as breaths.
5) Final Duration Threshold: A breath sound is typically longer than 100 ms. Therefore, if the detected breath event, 
after accurate edge marking, is shorter than that duration, it should be rejected. In practice, in order to account for 
very short breaths as well, the duration threshold may be set more permissively.
'''
# todo - ai: tune these parameters
preliminaryDurationThreshold = 0
UpperEnergyThreshold = 0
LowerZCRThreshold = 0
UpperZCRThreshold = 0
FinalDurationThreshold = 0

'''
Section III-B : Edge Detection
'''
'''
Section III-B.1: Edge Marking Using Double Energy Threshold and Deep Picking
'''
# todo - ai : very simple but requires fine tuning. Implement later on.

'''
Section III-B.2: Edge Marking With Spurious Deep Elimination
'''
# The binary vector of breathiness indices
breathinessIndices = []
for frame in initialClassifications:
    breathinessIndices.append(frame['breathinessIndice'])

# To reduce the effect of possible false detections, the binary vector of breathiness indices is first smoothed with a
# nine-point median filter.
breathinessIndices = sp.medfilt(breathinessIndices, 9)
# The block of ones indicates the approximate location of the breath, and the algorithm will look for the exact edges
# in the vicinity of this block.
# print(breathnessIndices)
# Let us denote the first frame index (representing its location along the time axis) of the block of ones as Xb1 and
# its last frame index as Xb2. For simplicity, we shall refer to the section [Xb1, Xb2] as the “candidate section.”
leftFound = False
leftIdx = 0
candidateSections = []
for idx in range(len(breathinessIndices)):
    if 1 == breathinessIndices[idx] and (not leftFound):
        leftIdx = idx
        leftFound = True
    elif 0 == breathinessIndices[idx] and leftFound:
        candidateSections.append({'startIdx': leftIdx, 'endIdx': idx - 1})
        leftFound = False  # reset bool as we go to new block
# print(candidateSections)
energyContours = []
energyContourOfSection = []
for section in candidateSections:
    # The edge search is conducted by examining the section’s energy contour
    energyContourOfSection.clear()
    for frame in initialClassifications[section['startIdx']:section['endIdx'] + 1]:
        # +1 is added above because we want to include energy of the last secion too.
        energyContourOfSection.append(frame['shortTimeEnergy'])
    # To reduce the number of such deeps, the energy contour is prefiltered with a threepoint running average filter
    energyContourOfSection = utils.calcRunningAvg(energyContourOfSection, 3)
    # plt.plot(energyContourOfSection)
    # plt.show()
    # After prefiltering, the remaining deeps are divided into significant and insignificant
    # todo - ai : continue here
# print(candidateSections)

import os
import scipy.io.wavfile
import numpy as np
import utils

folderBreaths = 'E:/atili/Datasets/BreathDataset/Datatset20191023/breaths_02-10_All/'

print('Controlling Stationarity at: %s' % folderBreaths)
totalP = 0
totalAdf = 0
totalAdf001 = 0
totalAdf005 = 0
totalAdf010 = 0
cntSample = 0
pVal001Stationary = 0
pVal001NonStationary = 0
pVal005Stationary = 0
pVal005NonStationary = 0
pVal010Stationary = 0
pVal010NonStationary = 0
adf001Stationary = 0
adf001NonStationary = 0
adf005Stationary = 0
adf005NonStationary = 0
adf010Stationary = 0
adf010NonStationary = 0
adfs = list()
pVals = list()
adfsWei = list()
pValsWei = list()

for rootPath, directories, files in os.walk(folderBreaths):
    np.random.shuffle(files)
    cntSample = len(files)
    ii = 0
    for filename in files:
        if '.wav' in filename:
            ii += 1
            print()
            print('%d of %d - Examining %s ...' % (ii, cntSample, filename))
            filepath = rootPath + filename
            samplingRate, audioData = scipy.io.wavfile.read(filepath)

            if 1 < len(audioData.shape):
                audioData = audioData[0]

            pVal, adfStat, adfCrit = utils.isStationary(audioData)
            pVals.append(pVal)
            adfs.append(adfStat)
            if pVal >= 0.01:
                pVal001NonStationary += 1
            else:
                pVal001Stationary += 1

            if pVal >= 0.05:
                pVal005NonStationary += 1
            else:
                pVal005Stationary += 1

            if pVal >= 0.10:
                pVal010NonStationary += 1
            else:
                pVal010Stationary += 1

            if adfStat >= adfCrit['1%']:
                adf001NonStationary += 1
            else:
                adf001Stationary += 1

            if adfStat >= adfCrit['5%']:
                adf005NonStationary += 1
            else:
                adf005Stationary += 1

            if adfStat >= adfCrit['10%']:
                adf010NonStationary += 1
            else:
                adf010Stationary += 1

            totalAdf001 += adfCrit['1%']
            totalAdf005 += adfCrit['5%']
            totalAdf010 += adfCrit['10%']
            totalAdf += adfStat
            totalP += pVal

avgP = totalP / cntSample
avgAdf = totalAdf / cntSample
avgAdf001 = totalAdf001 / cntSample
avgAdf005 = totalAdf005 / cntSample
avgAdf010 = totalAdf010 / cntSample

adfsBase = np.array([i * 100 for i in adfs], dtype=np.int)
pValsBase = np.array([i * 100 for i in pVals], dtype=np.int)

for i in adfs:
    adfsWei.append(np.sum(int(i*100) == adfsBase))
for i in pVals:
    pValsWei.append(np.sum(int(i*100) == pValsBase))

print('Adf Avg:', avgAdf)
print('Adf Wei Avg:', np.average(adfs, weights=adfsWei))
print('Adf Avg 0.01:', avgAdf001)
print('Adf Avg 0.05:', avgAdf005)
print('Adf Avg 0.10:', avgAdf010)
print('PVal Avg:', avgP)
print('PVal Wei Avg:', np.average(pVals, weights=pValsWei))
print('pVal001Stationary:', pVal001Stationary)
print('pVal001NonStationary:', pVal001NonStationary)
print(pVal001Stationary/(pVal001Stationary+pVal001NonStationary))
print('pVal005Stationary:', pVal005Stationary)
print('pVal005NonStationary:', pVal005NonStationary)
print(pVal005Stationary/(pVal005Stationary+pVal005NonStationary))
print('pVal010Stationary:', pVal010Stationary)
print('pVal010NonStationary:', pVal010NonStationary)
print(pVal010Stationary/(pVal010Stationary+pVal010NonStationary))
print('adf001Stationary:', adf001Stationary)
print('adf001NonStationary:', adf001NonStationary)
print(adf001Stationary/(adf001Stationary+adf001NonStationary))
print('adf005Stationary:', adf005Stationary)
print('adf005NonStationary:', adf005NonStationary)
print(adf005Stationary/(adf005Stationary+adf005NonStationary))
print('adf010Stationary:', adf010Stationary)
print('adf010NonStationary:', adf010NonStationary)
print(adf010Stationary/(adf010Stationary+adf010NonStationary))

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

paths = [
    'E:/atili/Datasets/BreathDataset/Datatset20191023/breaths_02-10_1',
    'E:/atili/Datasets/BreathDataset/Datatset20191023/breaths_02-10_2'
         ]

lengths = list()
for filepath in paths:
    for root, directories, files in os.walk(filepath):
        for file in files:
            if '.wav' in file:
                # print('Extracting Breaths of:', file, 'at', root)
                length = file.split('.')[2]
                length = int(length)/10
                lengths.append(length)
                # print(file, length)

print(max(lengths))
print(len(lengths))
# stats = np.zeros(int(max(lengths))+1)
# for l in lengths:
#     stats[int(l)] += 1

lowerBound = min(lengths)
higherBound = max(lengths)
mean = np.mean(lengths)
std = np.std(lengths)
a, b = (lowerBound - mean) / std, (higherBound - mean) / std

myDist = st.truncnorm(a=a, b=b, loc=mean, scale=std)

x = range(0, 1000, 1)
# plt.plot(x, myDist.pdf(x), label='Truncnorm')
plt.hist(lengths, bins=range(200, 1001, 25), range=(200, 1000), normed=False, label='Breath Instances')

plt.xticks(range(0, 1001, 100))

step = 50
start = 200
for minLen in range(start, int(np.ceil(max(lengths))), step):
    maxLen = minLen+step
    print(minLen, '-', maxLen, ':', sum(minLen <= j < maxLen for j in lengths) / len(lengths))
print('200ms <= i < 500ms :', sum(200 <= i < 500 for i in lengths) / len(lengths))
print('200ms <= i < 550ms :', sum(200 <= i < 550 for i in lengths) / len(lengths))
print('200ms <= i < 600ms :', sum(200 <= i < 600 for i in lengths) / len(lengths))
print('200ms <= i < 650ms :', sum(200 <= i < 650 for i in lengths) / len(lengths))

selected = list()
for i in lengths:
    if i <= 500:
        selected.append(i)
mean = np.mean(selected)
print(mean)

cutOffAmounts = 0
numOfLongRecs = 0
for i in lengths:
    if i > 600:
        cutOffAmounts += i-600
        numOfLongRecs += 1
print(cutOffAmounts/numOfLongRecs)

plt.vlines([200, 600], 0, 600, linestyles='--', colors='orange', label='90% of Data')
plt.ylabel('Number of Breath Instances')
plt.xlabel('Duration Bins [25 ms]')
# plt.vlines([mean], 0, 0.004, linestyles='--', colors='green', label='Mean')
plt.legend()
plt.savefig('./plots/lengthAnalysis.svg')
plt.savefig('./plots/lengthAnalysis.png')
plt.show()




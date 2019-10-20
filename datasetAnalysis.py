import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

paths = [
    'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10_1',
    'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10_2'
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
plt.plot(x, myDist.pdf(x), label='Truncnorm')
plt.hist(lengths, bins=range(0, 1001, 50), range=(0, 1000), normed=True, label='Data')

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

plt.vlines([200, 500], 0, 0.004, linestyles='--', colors='blue', label='Selection')
plt.vlines([mean], 0, 0.004, linestyles='--', colors='green', label='Mean')
plt.legend()
plt.show()




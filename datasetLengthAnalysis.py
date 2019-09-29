import os
import numpy as np
import matplotlib.pyplot as plt

filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/max_sample_set/wav/'

lengths = list()
for root, directories, files in os.walk(filepath):
    for file in files:
        if '.wav' in file:
            # print('Extracting Breaths of:', file, 'at', root)
            length = file.split('.')[2]
            while len(length) < 3:
                length += '0'
            length = int(length)
            lengths.append(length)
            print(file, length)

print(max(lengths))
stats = np.zeros(max(lengths)+1)
for l in lengths:
    stats[l] += 1

# plt.scatter(range(len(stats)), stats)
plt.plot(stats, linewidth=0.8)
plt.show()

step = 50
start = 200
for minLen in range(start, max(lengths), step):
    maxLen = minLen+step
    print(minLen, '-', maxLen, ':', sum(minLen <= j < maxLen for j in lengths) / len(lengths))
print('200 <= i < 500 :', sum(200 <= i < 500 for i in lengths) / len(lengths))
print('200 <= i < 550 :', sum(200 <= i < 550 for i in lengths) / len(lengths))
print('200 <= i < 600 :', sum(200 <= i < 600 for i in lengths) / len(lengths))
print('200 <= i < 650 :', sum(200 <= i < 650 for i in lengths) / len(lengths))

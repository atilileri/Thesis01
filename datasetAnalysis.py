import os
import numpy as np
import matplotlib.pyplot as plt

paths = [
    'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10'
    # , 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_00-02'
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
                print(file, length)

print(max(lengths))
stats = np.zeros(int(max(lengths))+1)
for l in lengths:
    stats[int(l)] += 1

# plt.scatter(range(len(stats)), stats)
plt.plot(stats, linewidth=0.8)
plt.show()

step = 50
start = 200
for minLen in range(start, int(np.ceil(max(lengths))), step):
    maxLen = minLen+step
    print(minLen, '-', maxLen, ':', sum(minLen <= j < maxLen for j in lengths) / len(lengths))
print('200ms <= i < 500ms :', sum(200 <= i < 500 for i in lengths) / len(lengths))
print('200ms <= i < 550ms :', sum(200 <= i < 550 for i in lengths) / len(lengths))
print('200ms <= i < 600ms :', sum(200 <= i < 600 for i in lengths) / len(lengths))
print('200ms <= i < 650ms :', sum(200 <= i < 650 for i in lengths) / len(lengths))

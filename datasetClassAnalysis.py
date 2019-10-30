import os
import numpy as np
import matplotlib.pyplot as plt

paths = [
    'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10_1'
    # , 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/breaths_02-10_2'
         ]

classes = dict()
lengths = list()
totalFileCount = 0
for filepath in paths:
    for root, directories, files in os.walk(filepath):
        for file in files:
            if '.wav' in file:
                # print('Extracting Breaths of:', file, 'at', root)
                speakerName = file[0:2]
                postureNo = file[2:4]
                if speakerName in classes:
                    if postureNo in classes[speakerName]:
                        classes[speakerName][postureNo] += 1
                    else:
                        classes[speakerName][postureNo] = 1
                else:
                    classes[speakerName] = dict()
                    classes[speakerName][postureNo] = 1
                totalFileCount += 1
print(classes)
avgFileCount = totalFileCount / len(classes)
print(avgFileCount)

objects = list(classes.keys())
y_pos = np.arange(len(objects))
bottoms = [0] * len(objects)
for posture in ['01', '02', '03', '04', '05']:
    breathCount = list()
    for speaker in classes:
        breathCount.append(classes[speaker][posture])
        # plt.bar
    plt.bar(y_pos, breathCount, align='center', alpha=0.8, bottom=bottoms, label=posture)
    bottoms = [sum(x) for x in zip(breathCount, bottoms)]

plt.xticks(y_pos, objects)
plt.hlines(np.mean(avgFileCount), -1, 20)
plt.ylabel('Breath Intances')
plt.title('Speaker Class Analysis')
plt.legend()
plt.show()

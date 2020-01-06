import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

paths = [
    'E:/atili/Datasets/BreathDataset/Processed/Inputs_Max_20191020/inputsFrom_02-10_1'
    , 'E:/atili/Datasets/BreathDataset/Processed/Inputs_Max_20191020/inputsFrom_02-10_2'
         ]

classes = dict()
lengths = list()
totalFileCount = 0
for filepath in paths:
    for root, directories, files in os.walk(filepath):
        for file in files:
            if '.imf48' in file:
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
                    classes[speakerName]['files'] = list()
                    classes[speakerName][postureNo] = 1
                classes[speakerName]['files'].append(root+'/'+file)
                totalFileCount += 1

minFileCount = totalFileCount
maxFileCount = 0
for speaker in classes:
    # shuffle files
    np.random.shuffle(classes[speaker]['files'])
    # update stats
    minFileCount = min(len(classes[speaker]['files']), minFileCount)
    maxFileCount = max(len(classes[speaker]['files']), maxFileCount)
avgFileCount = totalFileCount / len(classes)

print(classes)
print(minFileCount)
print(avgFileCount)
print(maxFileCount)

objects = list(classes.keys())
y_pos = np.arange(len(objects))
bottoms = [0] * len(objects)
postureNames = ['Sitting', 'Low Sitting', 'Standing', 'Hands Behind', 'Lying']
for posture in ['01', '02', '03', '04', '05']:
    breathCount = list()
    for speaker in classes:
        breathCount.append(classes[speaker][posture])
        # plt.bar
    plt.bar(y_pos, breathCount, align='center', alpha=0.8, bottom=bottoms, label=postureNames[int(posture)-1])
    bottoms = [sum(x) for x in zip(breathCount, bottoms)]

plt.xticks(y_pos, objects)
plt.hlines([minFileCount, avgFileCount, maxFileCount], -1, 20, linestyles=['dotted', 'dashed', 'dotted'])
plt.ylabel('Number of Breath Intances')
plt.xlabel('Speakers')
# plt.title('Speaker Class Analysis')
plt.legend()
plt.savefig('./plots/classAnalysis.svg')
plt.savefig('./plots/classAnalysis.png')
plt.show()

# USE WITH CAUTION: Copy up to maxFileCountPerPerson files from each person, for smaller dataset creation
# maxFileCountPerPerson = 100
# for speaker in classes:
#     print('Copying Speaker', speaker, '...')
#     files = classes[speaker]['files'][:maxFileCountPerPerson]
#     for i in range(len(files)):
#         src = files[i]
#         dst = src.replace('inputsFrom_', 'allSmall_', 1)
#         print(i+1, '- From:', src)
#         if not os.path.exists(os.path.dirname(dst)):
#             os.makedirs(os.path.dirname(dst))
#         shutil.copy(src, dst)
#         print('Copied To:', dst)
#         i += 1

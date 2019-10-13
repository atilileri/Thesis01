import scipy.io.wavfile
import os
import numpy as np
import scipy.signal
import gc


def filterRecording(folder, fileName):
    print('Filtering', fileName, 'at', folder, flush=True)
    fs, in4ch = scipy.io.wavfile.read(folder + '/' + fileName)
    in4ch = in4ch.view()  # create a view for copy comparison
    inBase = in4ch.base

    in4ch = np.swapaxes(in4ch, 0, 1)
    assert in4ch.base is inBase  # check for copy. we can not copy, no memory waste!

    out4ch = np.zeros_like(in4ch).view()  # create a view for copy comparison
    outBase = out4ch.base

    print('Filtering...', flush=True)
    filterSize = 4097
    h = scipy.signal.firwin(numtaps=filterSize, cutoff=70, pass_zero=False, fs=48000)
    for chIdx in range(len(in4ch)):
        np.put(out4ch[chIdx], range(in4ch[chIdx].size), scipy.signal.filtfilt(h, 1, in4ch[chIdx]))
    assert out4ch.base is outBase

    print('Saving...', flush=True)
    # restore channel index order
    in4ch = np.swapaxes(np.asarray(in4ch, dtype=np.float32), 0, 1)
    assert in4ch.base is inBase

    out4ch = np.swapaxes(np.asarray(out4ch, dtype=np.float32), 0, 1)
    assert out4ch.base is outBase

    fileSaveName = '_filt.'.join(fileName.split('.'))
    print(fileSaveName, 'saved to', folder)
    scipy.io.wavfile.write(folder + '/' + fileSaveName, fs, out4ch)
    # print(np.shape(out4ch))
    del out4ch
    fileSaveName = '_orig.'.join(fileName.split('.'))
    print(fileSaveName, 'saved to', folder)
    scipy.io.wavfile.write(folder + '/' + fileSaveName, fs, in4ch)
    # print(np.shape(in4ch))
    del in4ch
    gc.collect()


# subfolder = 'ce/'
subfolder = 'sd2/'
folderpath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/Recordings_Max/' + subfolder
# folderpath = 'D:/'

for root, directories, files in os.walk(folderpath):
    for file in files:
        if ('.wav' in file) and ('_orig.wav' not in file) and ('_filt.wav' not in file):
            filterRecording(root, file)

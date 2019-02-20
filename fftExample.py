import matplotlib.pyplot as plt
import numpy as np
import os
import utils
import sounddevice as sd
import time

# filepath = './TED Recordings/BillGates_2009/BillGates_2009.wav'
# filepath = './TED Recordings/BillGates_2009/bg/BillGates_2009_00.55.000-571_bg.wav'
# filepath = './TED Recordings/BillGates_2009/breath/BillGates_2009_00.32.850-525_breath.wav'
# filepath = './TED Recordings/BillGates_2009/non-voiced/BillGates_2009_00.21.576-326_nv.wav'
# filepath = './TED Recordings/BillGates_2009/speech/BillGates_2009_01.32.610-614_spch.wav'
# filepath = './METU Recordings/hh2_bg/hh2_00.09.569-2266_bg.wav'
# filepath = './METU Recordings/hh2_breath/hh2_08_01.05.315_246_en.wav'
filepath = './METU Recordings/hh2_speech/hh2_01.28.155-896_sp.wav'

sampRate, sig = utils.readMonoWav(filepath)
filename = str(os.path.basename(filepath).rsplit('.', 1)[0])

sp = np.fft.fft(sig)
freq = np.fft.fftfreq(len(sig))
plt.subplot(311)
plt.title('Signal: ' + filename)
plt.plot(sig)
plt.subplot(312)
plt.title('FFT Real')
# for fr in freq:
#     print(fr)
plt.plot(freq[:int(len(freq)/2)], sp.real[:int(len(sp.real)/2)])
plt.subplot(313)
plt.title('FFT Imaginary')
plt.plot(freq[:int(len(freq)/2)], sp.imag[:int(len(sp.imag)/2)])
plt.show()

sd.play(sig, sampRate)
time.sleep(len(sig)/sampRate)
sd.stop()

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

cutoff = 55
fs = 48000
filterLengths = [(np.power(2, i)*512) + 1 for i in range(0, 8)]
for fl in filterLengths:
    print('Drawing Filter Length: ', fl, flush=True)
    b = signal.firwin(numtaps=fl, cutoff=cutoff, pass_zero=False, fs=fs)
    w, h = signal.freqz(b)

    plt.title('Digital filter frequency response (cutoff: %d Hz)' % cutoff)
    # w is returned in rad/sample. Convert to Hz by:  w * (1/(2*np.pi)) * fs
    # h is returned in SPL. Convert to dB by: 20 * np.log10(abs(h))
    # draw only first few samples for detailed view
    plt.plot(w[0:5]*(1/(2*np.pi))*fs, 20*np.log10(abs(h[0:5])), label="filterLen: %d" % fl)
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.yticks([0, -3, -10, -20, -30, -40, -50, -80])
plt.xticks([0, 25, 50, 55, 60, 70, 80, 90, 95, 100, 150, 190], rotation='vertical')
plt.grid(linestyle=':')
plt.legend()
plt.axis('tight')
fig = plt.gcf()
plt.show()
fig.set_size_inches(40, 40)
fig.savefig('./plots/filter.png')

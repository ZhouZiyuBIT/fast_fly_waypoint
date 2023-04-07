import numpy as np
import matplotlib.pyplot as plt

# from scipy.fft import fft, ifft
def fft_filter(siginal_orig, freq_cut, vis=False):

    mean = np.mean(siginal_orig)
    siginal_orig_ = siginal_orig - mean
    fft_signal = np.fft.fft(siginal_orig_)
    freqs = np.fft.fftfreq(len(siginal_orig_), d=0.01)

    fft_signal_filted = fft_signal.copy()
    mask = (freqs > freq_cut) + (freqs < -freq_cut)
    fft_signal_filted[mask] = 0
    siginal_filtered = np.real(np.fft.ifft(fft_signal_filted))
    siginal_filtered += mean

    if vis:
        plt.figure("Freq. Plot")
        plt.stem(freqs, np.abs(fft_signal), basefmt=None, markerfmt='.')
        plt.figure("Signal Plot")
        plt.plot(siginal_orig, label='Original Signal')
        plt.plot(siginal_filtered, label='Filtered Signal')
        plt.legend()

    return siginal_filtered

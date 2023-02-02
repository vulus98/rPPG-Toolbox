""" Short functions for data-preprocessing and data-loading. """

import numpy as np
import cv2


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)

# Generating noise for covering original pulse rate signal
def fftnoise(f):
    np.random.seed()
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def noise_signal_generator(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def sine_wave(freq,samples):
    l=samples/30
    x=np.linspace(0,l,samples)
    y=np.sin(2*np.pi*freq/60*x)
    return y



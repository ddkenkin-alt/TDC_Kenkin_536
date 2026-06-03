import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import os

n = 500
Fs = 1000
F_max = 15

random_signal = np.random.normal(0, 10, n)

t = np.arange(n) / Fs

w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')

filtered_signal = signal.sosfiltfilt(sos, random_signal)

def plot_signal(x, y, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))

    ax.plot(x, y, linewidth=1)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)

    ax.tick_params(labelsize=12)

    ax.grid(True)

    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/{filename}.png", dpi=600)

    plt.close(fig)

plot_signal(
    t,
    filtered_signal,
    "Filtered signal (F_max = 15 Hz)",
    "Time [s]",
    "Amplitude",
    "signal"
    )

spectrum = fft.fft(filtered_signal)
spectrum = np.abs(fft.fftshift(spectrum))

freqs = fft.fftfreq(n, 1 / Fs)
freqs = fft.fftshift(freqs)

plot_signal(
    freqs,
    spectrum,
    "Signal spectrum",
    "Frequency [Hz]",
    "Magnitude",
    "spectrum")
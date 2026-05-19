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

# ЛР3 -------------------------------------------------------------------------

Fs = 1000
F_filter = 22
n = 500

signal_base = filtered_signal

discrete_signals = []
discrete_spectrums = []
reconstructed_signals = []

noise_variances = []
snr_values = []

Dt_list = [2, 4, 8, 16]

for Dt in Dt_list:

    discrete_signal = np.zeros(n)

    for i in range(0, n // Dt):
        discrete_signal[i * Dt] = signal_base[i * Dt]

    discrete_signals.append(discrete_signal)

    spectrum = fft.fft(discrete_signal)
    spectrum = np.abs(fft.fftshift(spectrum))
    discrete_spectrums.append(spectrum)

    w = F_filter / (Fs / 2)
    sos = signal.butter(3, w, 'low', output='sos')
    restored = signal.sosfiltfilt(sos, discrete_signal)

    reconstructed_signals.append(restored)

    error = restored - signal_base

    var_signal = np.var(signal_base)
    var_error = np.var(error)

    noise_variances.append(var_error)
    snr_values.append(var_signal / var_error)
    
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

s = 0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(discrete_signals[s], linewidth=1)
        ax[i][j].set_title(f"Dt = {Dt_list[s]}", fontsize=14)
        ax[i][j].grid(True)
        s += 1

fig.suptitle("Discrete signals", fontsize=14)
fig.supxlabel("Samples", fontsize=14)
fig.supylabel("Amplitude", fontsize=14)

fig.savefig("./figures/discrete_signals.png", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

freqs = fft.fftshift(fft.fftfreq(n, 1/Fs))

s = 0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(freqs, discrete_spectrums[s], linewidth=1)
        ax[i][j].set_title(f"Spectrum Dt = {Dt_list[s]}", fontsize=14)
        ax[i][j].grid(True)
        s += 1

fig.suptitle("Spectra of discrete signals", fontsize=14)
fig.supxlabel("Frequency (Hz)", fontsize=14)
fig.supylabel("Magnitude", fontsize=14)

fig.savefig("./figures/discrete_spectra.png", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

t = np.arange(n) / Fs

s = 0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(t, reconstructed_signals[s], linewidth=1)
        ax[i][j].set_title(f"Restored Dt = {Dt_list[s]}", fontsize=14)
        ax[i][j].grid(True)
        s += 1

fig.suptitle("Reconstructed signals", fontsize=14)
fig.supxlabel("Time (s)", fontsize=14)
fig.supylabel("Amplitude", fontsize=14)

fig.savefig("./figures/reconstructed.png", dpi=600)
plt.close(fig)

plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot(Dt_list, noise_variances, marker='o')
plt.grid(True)
plt.title("Noise variance vs sampling step", fontsize=14)
plt.xlabel("Dt", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.savefig("./figures/variance.png", dpi=600)
plt.close()

plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot(Dt_list, snr_values, marker='o')
plt.grid(True)
plt.title("SNR vs sampling step", fontsize=14)
plt.xlabel("Dt", fontsize=14)
plt.ylabel("SNR", fontsize=14)
plt.savefig("./figures/snr.png", dpi=600)
plt.close()
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
from skimage.restoration import cycle_spin

import pywt

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"


def wavelet_denoiser(signal, level=5, mode='hard', wavelet='db4'):

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode)
        for c in coeffs[1:]
    ]

    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal[:len(signal)]


def gaussian_kernel(size, sigma):

    x = np.linspace(-(size // 2), size // 2, size)

    kernel = np.exp(-0.5 * (x / sigma) ** 2)

    return kernel / kernel.sum()


def wavelet_shifted_filter():

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)

    if len(data.shape) > 1:
        data = data[:, 0]

    time = np.arange(len(data)) / fs_original

    max_shifts = [0, 1, 3, 5]

    signals = []

    for n, s in enumerate(max_shifts):

        sig_filtered = cycle_spin(
            data,
            func=wavelet_denoiser,
            max_shifts=s,
            shift_steps=5
        )

        sf.write(
            f"./Sounds/Filtered_Shifted_Wavelet_{n}.wav",
            sig_filtered,
            SAMPLE_RATE
        )

        signals.append(sig_filtered)

    kernel = gaussian_kernel(size=11, sigma=2)

    filtered_signal = convolve(
        data,
        kernel,
        mode='same'
    )

    sf.write(
        "./Sounds/Filtered_Gaussian_Filter.wav",
        filtered_signal,
        SAMPLE_RATE
    )

    plt.figure(figsize=(12, 6))

    plt.plot(time, data,
             label=f"Оригінал (fs={SAMPLE_RATE} Гц)")

    plt.plot(time, signals[0],
             label="Wavelet Shifted: no shift")

    plt.plot(time, signals[1],
             label="Wavelet Shifted: 1x2")

    plt.plot(time, signals[2],
             label="Wavelet Shifted: 1x4")

    plt.plot(time, signals[3],
             label="Wavelet Shifted: 1x6")

    plt.title(
        "Порівняння сигналів у часовій області, вейвлет-фільтр, модифікований"
    )

    plt.xlabel("Час (с)")
    plt.ylabel("Амплітуда")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Wavelet_Shifted_Comparison.png")

    plt.show()

    plt.figure(figsize=(12, 6))

    plt.plot(time, data,
             label=f"Оригінал (fs={SAMPLE_RATE} Гц)")

    plt.plot(time,
             filtered_signal,
             label="Gaussian Filter")

    plt.title(
        "Порівняння сигналів у часовій області, фільтр Гаусса"
    )

    plt.xlabel("Час (с)")
    plt.ylabel("Амплітуда")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Gaussian_Filter_Comparison.png")

    plt.show()


if __name__ == "__main__":
    wavelet_shifted_filter()
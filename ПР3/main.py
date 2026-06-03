import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import (
    denoise_wavelet,
    denoise_invariant,
    denoise_tv_chambolle,
    denoise_bilateral
)

import pywt

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"

NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = "./Sounds/Sound_4000[Hz]_2[byte].raw"

NAME_FILTERED_WAV = "./Sounds/Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = "./Sounds/Filtered_4000[Hz]_2[byte].raw"

def wavelet_denoiser(signal, level, mode, wavelet):

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode)
        for c in coeffs[1:]
    ]

    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal[:len(signal)]


def invarince_denoiser(image, **kwargs):
    return denoise_wavelet(
        image,
        sigma=0.5,
        wavelet='db4',
        mode='soft'
    )


def sound_filter():

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)

    if len(data.shape) > 1:
        data = data[:, 0]

    time = np.arange(len(data)) / fs_original

    data_2d = data.reshape(1, -1)

    invariance = denoise_invariant(
        data_2d,
        denoise_function=invarince_denoiser
    ).flatten()

    total_variation = denoise_tv_chambolle(
        data_2d,
        weight=0.1,
        channel_axis=None
    ).flatten()

    bilateral = denoise_bilateral(
        data_2d,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=None
    ).flatten()

    wavelet = wavelet_denoiser(
        data,
        level=5,
        mode='soft',
        wavelet='db4'
    )

    sf.write("./Sounds/Filtered_Invariance.wav",
             invariance, SAMPLE_RATE)

    sf.write("./Sounds/Filtered_Total_Variation.wav",
             total_variation, SAMPLE_RATE)

    sf.write("./Sounds/Filtered_Bilateral.wav",
             bilateral, SAMPLE_RATE)

    sf.write("./Sounds/Filtered_Wavelet.wav",
             wavelet, SAMPLE_RATE)

    filters = [
        ("J-Invariance", invariance, "Filtered_J_Invariance.png"),
        ("Total Variation", total_variation, "Filtered_TV.png"),
        ("Bilateral", bilateral, "Filtered_Bilateral.png"),
        ("Wavelet", wavelet, "Filtered_Wavelet.png")
    ]

    for title, filtered_signal, filename in filters:

        plt.figure(figsize=(10, 6))

        plt.plot(time, data,
                 label="Original Signal")

        plt.plot(time, filtered_signal,
                 linewidth=2,
                 label=title)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.savefig(filename)
        plt.show()


if __name__ == "__main__":
    sound_filter()
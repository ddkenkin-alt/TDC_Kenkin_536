import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import glob

from scipy.signal import resample
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from skimage.restoration import (
    denoise_wavelet
)

import pywt


SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

NAME_ORIGINAL_WAV = (
    f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
)

NAME_ORIGINAL_RAW = (
    f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"
)

NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"

NAME_RESAMPLED_RAW = "./Sounds/Sound_4000[Hz]_2[byte].raw"

NAME_FILTERED_WAV = "./Sounds/Filtered_4000[Hz]_2[byte].wav"

NAME_FILTERED_RAW = "./Sounds/Filtered_4000[Hz]_2[byte].raw"


def wavelet_denoiser(
        signal,
        level=5,
        mode='hard',
        wavelet='db4'
):

    coeffs = pywt.wavedec(
        signal,
        wavelet,
        level=level
    )

    sigma = np.median(
        np.abs(coeffs[-1])
    ) / 0.6745

    threshold = (
        sigma *
        np.sqrt(
            2 * np.log(signal.size)
        )
    )

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(
            c,
            threshold,
            mode=mode
        )
        for c in coeffs[1:]
    ]

    denoised_signal = pywt.waverec(
        denoised_coeffs,
        wavelet
    )

    return denoised_signal[:len(signal)]


def gaussian_kernel(size, sigma):

    x = np.linspace(
        -(size // 2),
        size // 2,
        size
    )

    kernel = np.exp(
        -0.5 * (x / sigma) ** 2
    )

    return kernel / kernel.sum()


def invariance_denoiser(image, **kwargs):

    return denoise_wavelet(
        image,
        sigma=0.5,
        wavelet='db4',
        mode='soft'
    )


def to_scientific_pretty(
        x,
        precision=2
):

    superscripts = str.maketrans(
        "0123456789-",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
    )

    mantissa, exponent = (
        f"{x:.{precision}e}"
        .split('e')
    )

    mantissa = (
        mantissa
        .rstrip('0')
        .rstrip('.')
    )

    return (
        f"{mantissa} · 10"
        f"{str(int(exponent)).translate(superscripts)}"
    )


if __name__ == "__main__":

    results = []

    row = []

    headers = [
        "MSE",
        "MAE",
        "RMSE",
        "R2",
        "D"
    ]

    data_original, fs_original = sf.read(
        NAME_ORIGINAL_WAV
    )

    if len(data_original.shape) > 1:
        data_original = data_original[:, 0]

    wav_files = glob.glob(
        "./Sounds/*.wav"
    )

    for sounds in wav_files:

        sounds = sounds.replace(
            "\\",
            "/"
        )

        if sounds == NAME_ORIGINAL_WAV:
            continue

        elif sounds == NAME_RESAMPLED_WAV:

            row.append(
                "Ресемпл 4 кГц"
            )

            data, fs = sf.read(
                sounds
            )

            if len(data.shape) > 1:
                data = data[:, 0]

            data = resample(
                data,
                len(data_original)
            )

        else:

            type_filter = sounds.replace(
                "./Sounds/Filtered_",
                ""
            )

            type_filter = type_filter.replace(
                ".wav",
                ""
            )

            type_filter = type_filter.replace(
                "_",
                " "
            )

            if type_filter == "4000[Hz] 2[byte]":
                type_filter = (
                    "Лінійний фільтр 4 кГц"
                )

            row.append(
                type_filter
            )

            data, fs = sf.read(
                sounds
            )

            if len(data.shape) > 1:
                data = data[:, 0]

        mse = mean_squared_error(
            data_original,
            data
        )

        mae = mean_absolute_error(
            data_original,
            data
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            data_original,
            data
        )

        D = np.var(
            data_original - data
        )

        results.append([
            to_scientific_pretty(
                mse
            ),
            to_scientific_pretty(
                mae
            ),
            to_scientific_pretty(
                rmse
            ),
            round(
                r2,
                2
            ),
            to_scientific_pretty(
                D
            )
        ])

    n_rows = len(row)
    n_cols = len(headers)

    fig, ax = plt.subplots(
        figsize=(
            n_cols * 2.8,
            n_rows * 0.4
        )
    )

    ax.axis('off')

    table = ax.table(
        cellText=results,
        rowLabels=row,
        colLabels=headers,
        loc='center',
        bbox=[0.08, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.savefig(
        "Metrics_Table.png",
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()
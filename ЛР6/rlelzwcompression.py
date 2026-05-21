import ast
import math
import collections
import matplotlib.pyplot as plt


def encode_rle(sequence):
    count = 1
    result = []

    for i, item in enumerate(sequence):

        if i == 0:
            continue

        if item == sequence[i - 1]:
            count += 1

        else:
            result.append((sequence[i - 1], count))
            count = 1

    result.append((sequence[len(sequence) - 1], count))

    encoded = []

    for i, item in enumerate(result):
        encoded.append(f"{item[1]}{item[0]}")

    return "".join(encoded), result


def decode_rle(sequence):
    result = []

    for item in sequence:
        result.append(item[0] * item[1])

    return "".join(result)


def encode_lzw(sequence):

    dictionary = {}

    for i in range(65536):
        dictionary[chr(i)] = i

    current = ""
    result = []
    size = 0

    for c in sequence:

        new_str = current + c

        if new_str in dictionary:
            current = new_str

        else:

            result.append(dictionary[current])

            dictionary[new_str] = len(dictionary)

            element_bits = (
                16
                if dictionary[current] < 65536
                else math.ceil(math.log2(len(dictionary)))
            )

            with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:
                file.write(
                    f"Code: {dictionary[current]}, "
                    f"Element: {current}, "
                    f"Bits: {element_bits}\n"
                )

            size += element_bits

            current = c

    if current:

        last = (
            16
            if dictionary[current] < 65536
            else math.ceil(math.log2(len(dictionary)))
        )

        size += last

        with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:
            file.write(
                f"Code: {dictionary[current]}, "
                f"Element: {current}, "
                f"Bits: {last}\n"
            )

        result.append(dictionary[current])

    return result, size


def decode_lzw(sequence):

    dictionary = {}

    for i in range(65536):
        dictionary[i] = chr(i)

    result = ""
    previous = None
    current = ""

    for code in sequence:

        if code in dictionary:

            current = dictionary[code]
            result += current

            if previous is not None:
                dictionary[len(dictionary)] = previous + current[0]

            previous = current

        else:

            current = previous + previous[0]
            result += current

            dictionary[len(dictionary)] = current

            previous = current

    return result


N_sequence = 100


with open("sequence.txt", "r", encoding="utf-8") as file:

    original_sequences = ast.literal_eval(file.read())

    original_sequences = [
        sequence.strip("[]").strip("'").strip('"')
        for sequence in original_sequences
    ]


results = []

with open("results_rle_lzw.txt", "w", encoding="utf-8") as file:
    file.write("Результати стиснення RLE та LZW\n\n")


for index, sequence in enumerate(original_sequences):


    counts = collections.Counter(sequence)

    probability = {
        symbol: count / N_sequence
        for symbol, count in counts.items()
    }

    entropy = -sum(
        p * math.log2(p)
        for p in probability.values()
    )

    original_size = len(sequence) * 16


    encoded_sequence_rle, encoded_rle = encode_rle(sequence)

    encoded_rle_size = len(encoded_sequence_rle) * 16

    compression_ratio_rle = round(
        original_size / encoded_rle_size,
        2
    )

    if compression_ratio_rle < 1:
        compression_ratio_rle = '-'

    decoded_sequence_rle = decode_rle(encoded_rle)

    decoded_rle_size = len(decoded_sequence_rle) * 16


    with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:
        file.write("Кодування LZW\n")
        file.write("Словник\n")

    encoded_sequence_lzw, lzw_size = encode_lzw(sequence)

    compression_ratio_lzw = round(
        original_size / lzw_size,
        2
    )

    decoded_sequence_lzw = decode_lzw(encoded_sequence_lzw)

    decoded_lzw_size = len(decoded_sequence_lzw) * 16


    with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:

        file.write(
            f"\n"
            f"====================================================\n"
            f"Послідовність {index + 1}\n"
            f"====================================================\n"
        )

        file.write(
            f"Оригінальна послідовність: {sequence}\n"
        )

        file.write(
            f"Розмір оригінальної послідовності: "
            f"{original_size} bits\n"
        )

        file.write(
            f"Ентропія: {round(entropy, 4)}\n\n"
        )

        # RLE

        file.write("Кодування RLE\n")

        file.write(
            f"Закодована RLE послідовність: "
            f"{encoded_sequence_rle}\n"
        )

        file.write(
            f"Розмір закодованої RLE послідовності: "
            f"{encoded_rle_size} bits\n"
        )

        file.write(
            f"Коефіцієнт стиснення RLE: "
            f"{compression_ratio_rle}\n"
        )

        file.write(
            f"Декодована RLE послідовність: "
            f"{decoded_sequence_rle}\n"
        )

        file.write(
            f"Розмір декодованої RLE послідовності: "
            f"{decoded_rle_size} bits\n\n"
        )

        # LZW

        file.write(
            f"Закодована LZW послідовність: "
            f"{''.join(map(str, encoded_sequence_lzw))}\n"
        )

        file.write(
            f"Розмір закодованої LZW послідовності: "
            f"{lzw_size} bits\n"
        )

        file.write(
            f"Коефіцієнт стиснення LZW: "
            f"{compression_ratio_lzw}\n"
        )

        file.write(
            f"Декодована LZW послідовність: "
            f"{decoded_sequence_lzw}\n"
        )

        file.write(
            f"Розмір декодованої LZW послідовності: "
            f"{decoded_lzw_size} bits\n\n"
        )

    results.append([
        round(entropy, 2),
        compression_ratio_rle,
        compression_ratio_lzw
    ])



N = len(original_sequences)

fig, ax = plt.subplots(figsize=(14 / 1.54, N / 1.54))

headers = ['Ентропія', 'КС RLE', 'КС LZW']

rows = [
    f'Послідовність {i + 1}'
    for i in range(N)
]

ax.axis('off')

table = ax.table(
    cellText=results,
    colLabels=headers,
    rowLabels=rows,
    loc='center',
    cellLoc='center'
)

table.set_fontsize(14)
table.scale(0.8, 2)

fig.savefig(
    "Результати стиснення методами RLE та LZW.png",
    bbox_inches='tight'
)

print("Готово.")
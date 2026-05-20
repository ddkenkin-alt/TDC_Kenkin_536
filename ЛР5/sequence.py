import random
import string
import collections
import math
import matplotlib.pyplot as plt

N_sequence = 100

surname = "Кенкін"
group_number = "536"
student_number = 7

results = []
original_sequences = []


def save_sequence_info(file, name, sequence):
    sequence_alphabet_size = len(set(sequence))
    sequence_size = len(sequence)

    counts = collections.Counter(sequence)

    probability = {
        symbol: count / len(sequence)
        for symbol, count in counts.items()
    }

    mean_probability = (
        sum(probability.values()) / len(probability)
    )

    equal = all(
        abs(prob - mean_probability) < 0.05 * mean_probability
        for prob in probability.values()
    )

    if equal:
        uniformity = "рівна"
    else:
        uniformity = "нерівна"

    entropy = -sum(
        p * math.log2(p)
        for p in probability.values()
    )

    if sequence_alphabet_size > 1:
        source_excess = (
            1 - entropy / math.log2(sequence_alphabet_size)
        )
    else:
        source_excess = 1

    probability_str = ', '.join(
        [f"{symbol}={prob:.4f}" for symbol, prob in probability.items()]
    )

    file.write(f"{name}\n")
    file.write(f"Послідовність: {sequence}\n")
    file.write(f"Розмір послідовності {sequence_size} byte\n")
    file.write(f"Розмір алфавіту: {sequence_alphabet_size}\n")
    file.write(f"Ймовірності появи символів: {probability_str}\n")
    file.write(
        f"Середнє арифметичне ймовірностей: "
        f"{mean_probability:.4f}\n"
    )
    file.write(
        f"Ймовірність розподілу символів: "
        f"{uniformity}\n"
    )
    file.write(f"Ентропія: {entropy:.4f}\n")
    file.write(
        f"Надмірність джерела: "
        f"{source_excess:.4f}\n"
    )
    file.write("\n" + "=" * 80 + "\n\n")

    results.append([
        sequence_alphabet_size,
        round(entropy, 2),
        round(source_excess, 2),
        uniformity
    ])


# Послідовність 1 -------------------------------------------------------------

list1 = ['1'] * student_number
list0 = ['0'] * (N_sequence - student_number)

sequence_1 = list1 + list0
random.shuffle(sequence_1)

original_sequence_1 = ''.join(sequence_1)

original_sequences.append(original_sequence_1)

# Послідовність 2 -------------------------------------------------------------

list1 = list(surname)
list0 = ['0'] * (N_sequence - len(surname))

sequence_2 = list1 + list0

original_sequence_2 = ''.join(sequence_2)

original_sequences.append(original_sequence_2)

# Послідовність 3 -------------------------------------------------------------

list1 = list(surname)
list0 = ['0'] * (N_sequence - len(surname))

sequence_3 = list1 + list0
random.shuffle(sequence_3)

original_sequence_3 = ''.join(sequence_3)

original_sequences.append(original_sequence_3)

# Послідовність 4 -------------------------------------------------------------

letters = list(surname) + list(group_number)

n_letters = len(letters)

n_repeats = N_sequence // n_letters
remainder = N_sequence % n_letters

sequence_4 = letters * n_repeats
sequence_4 += letters[:remainder]

original_sequence_4 = ''.join(map(str, sequence_4))

original_sequences.append(original_sequence_4)

# Послідовність 5 -------------------------------------------------------------

elements = [
    surname[0],
    surname[1],
    group_number[0],
    group_number[1],
    group_number[2]
]

sequence_5 = []

for element in elements:
    sequence_5.extend([element] * 20)

random.shuffle(sequence_5)

original_sequence_5 = ''.join(sequence_5)

original_sequences.append(original_sequence_5)

# Послідовність 5 -------------------------------------------------------------

letters = [surname[0], surname[1]]
digits = list(group_number)

n_letters = int(0.7 * N_sequence)
n_digits = N_sequence - n_letters

sequence_6 = []

for _ in range(n_letters):
    sequence_6.append(random.choice(letters))

for _ in range(n_digits):
    sequence_6.append(random.choice(digits))

random.shuffle(sequence_6)

original_sequence_6 = ''.join(sequence_6)

original_sequences.append(original_sequence_6)

# Послідовність 7 -------------------------------------------------------------

elements = string.ascii_lowercase + string.digits

sequence_7 = [
    random.choice(elements)
    for _ in range(N_sequence)
]

original_sequence_7 = ''.join(sequence_7)

original_sequences.append(original_sequence_7)

# Послідовність 8 -------------------------------------------------------------

original_sequence_8 = '1' * N_sequence

original_sequences.append(original_sequence_8)


with open("LosslessCompression/sequence.txt", "w", encoding="utf-8") as file:

    for i, sequence in enumerate(original_sequences, start=1):
        file.write(f"Sequence {i}:\n")
        file.write(sequence + "\n\n")


with open(
    "LosslessCompression/results_sequence.txt",
    "w",
    encoding="utf-8"
) as file:

    for i, sequence in enumerate(original_sequences, start=1):

        save_sequence_info(
            file,
            f"Послідовність {i}",
            sequence
        )

headers = [
    'Розмір алфавіту',
    'Ентропія',
    'Надмірність',
    'Ймовірність'
]

rows = [
    f'Послідовність {i}'
    for i in range(1, 9)
]

fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))

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
    "LosslessCompression/Характеристики_сформованих_послідовностей.png",
    bbox_inches='tight'
)

plt.close()

print("Усі файли успішно створені!")
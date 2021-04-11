import itertools
import multiprocessing
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas
from tqdm import tqdm

N = 8
BASE = ["A", "T", "G", "C"]
N_BASE = ["".join(i) for i in list(itertools.product(BASE, repeat=N))]


def seq_to_odd(line):
    header, seq = line.split("\n")
    n_continuous_base_odd = calculate_odd_ratio(seq)
    return (
        header + "\n" + " ".join([str(n_continuous_base_odd[b]) for b in N_BASE]) + "\n"
    )


def calculate_odd_ratio(seq):
    n_continuous_base_odd = defaultdict(float)
    one_base = defaultdict(int)
    n_base = defaultdict(int)
    for i in range(len(seq)):
        one_base[seq[i]] += 1
        if i + N < len(seq):
            n_base[seq[i : i + N]] += 1

    total_n_continuous_base_freq = np.sum(list(n_base.values()))
    total_one_base_freq = np.sum(list(one_base.values()))

    for i in n_base.items():
        one_base_freq = 1
        for k in i[0]:
            one_base_freq *= one_base[k] / total_one_base_freq

        n_base_freq = i[1] / total_n_continuous_base_freq
        n_continuous_base_odd[i[0]] = "{:.6f}".format(n_base_freq / one_base_freq)
    return n_continuous_base_odd


def main():
    p = Pool(multiprocessing.cpu_count() - 1)
    seq_array = []
    filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(filename, "r", encoding="utf-8") as r:
        for line in tqdm(r.readlines()):
            if ">" in line:
                if "seq" in locals():
                    seq_array.append(seq)
                seq = line
            else:
                seq += line.strip()
        seq_array.append(seq)
    r.close()

    seq_array = p.map(seq_to_odd, tqdm(seq_array))

    with open(output_filename, "w", encoding="utf-8") as w:
        for line in tqdm(seq_array):
            w.write(line)
    w.close()


if __name__ in "__main__":
    main()

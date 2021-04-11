import pandas
import numpy as np
from collections import defaultdict
import itertools
from operator import mul
from functools import reduce
import sys
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

N = 5
BASE = ["A", "T", "G", "C"]
N_BASE = ["".join(i) for i in list(itertools.product(BASE, repeat=N))]

def calculate_odd_ratio(seq, n_continuous_base_odd):
    one_base = defaultdict(int)
    n_base = defaultdict(int)
    for j in range(len(seq)):
        one_base[seq[j]] += 1
        if j+N <= len(seq):
            n_base[seq[j:j+N]] += 1

    total_n_continuous_base_freq = np.sum([n_base[b] for b in N_BASE])
    total_one_base_freq = np.sum([one_base[b] for b in BASE])

    for j in n_base.items():
        one_base_freq = np.prod(
            [one_base[k] / total_one_base_freq for k in j[0]])
        n_base_freq = j[1]/total_n_continuous_base_freq
        n_continuous_base_odd[j[0]
                              ] = np.round(n_base_freq/one_base_freq, 6)
    return n_continuous_base_odd
    

if __name__ in "__main__":
    filename = sys.argv[1]
    output_filename = sys.argv[2]
    with open(filename, "r", encoding="utf-8") as r:
        with open(output_filename, "w", encoding="utf-8") as w:
            for line in tqdm(r.readlines()):
                if ">" in line:
                    if 'n_continuous_base_odd' in locals():
                        n_continuous_base_odd = calculate_odd_ratio(
                            seq, n_continuous_base_odd)
                        for j in N_BASE:
                            w.write(str(n_continuous_base_odd[j]) + " ")
                        w.write("\n")
                    seq = ""

                    n_continuous_base_odd = defaultdict(float)
                    w.write(line)
                else:
                    seq += line.strip()
            n_continuous_base_odd = calculate_odd_ratio(
                seq, n_continuous_base_odd)
            for j in N_BASE:
                w.write(str(n_continuous_base_odd[j]) + " ")
    r.close()
    w.close()

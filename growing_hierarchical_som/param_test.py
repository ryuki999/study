"""
param_test.py
param_test用のプログラム
"""

import os
import itertools
import pickle
import sys
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from ghsom_describe import *
from color_plot import interactive_plot_with_labels, image_plot_with_labels_save
import gc
from GHSOM import GHSOM

# モジュール検索パスに，ひとつ上の階層の絶対パスを追加
sys.path.append("..")
from create_dataset.fasta_to_df import all_data_df_to_arange_df, fasta_to_df


N = 5
BASE = ["A", "T", "G", "C"]
N_BASE = ["".join(i) for i in list(itertools.product(BASE, repeat=N))]
DATA_COLUMNS = ["year", "month", "day"]
CLADE_COLUMNS = ["clade", "head2"]

JAPAN_HEADER_COLUMNS = ["head", "ID", "date"]
ALL_DATA_HEADER_COLUMNS = [
    "head",
    "id",
    "continent",
    "country",
    "city",
    "host",
    "clade_head",
    "date",
]

def read_data(filename):
    with open(filename) as f:
        sample_genome = f.read().split("\n")
        sample_genome_array = []
        for i in range(2, len(sample_genome), 3):
            one_data = [int(float(n)) for n in sample_genome[i].split()]
            while len(one_data) != 144:
                one_data.append(0)
            sample_genome_array.append(one_data)
        sample_genome_df = pd.DataFrame(sample_genome_array)
    f.close()
    return sample_genome_df


if __name__ == "__main__":
    folder = sys.argv[1]
    df0 = read_data(f"{folder}Bacteria.frq")
    df0["label"] = 0
    df1 = read_data(f"{folder}Eukaryote.frq")
    df1["label"] = 1
    df2 = read_data(f"{folder}Virus.frq")
    df2["label"] = 2
    df_concat = pd.concat([df0, df1, df2])

    labels = np.array(df_concat["label"])
    data = np.array(df_concat.drop(columns=["label"]))
    n_samples, n_features = data.shape
    n_digits = len(np.unique(labels))

    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))
    t1 = [1, 0.1, 0.01, 0.001]
    t2 = [1, 0.1, 0.01, 0.001]
    gaussian_sigma = [2, 3, 4, 5]
    grow_maxiter = [1, 5, 10, 15, 20]
    epochs = [1,5,10,15]
    lr = 0.15
    decay = 0.95

    for t1_i in t1:
        for t2_i in t2:
            for gau_i in gaussian_sigma:
                for gr_i in grow_maxiter:
                    for ep in epochs:
                        start = time.time()
                        dir = f"t1c{t1_i}-t2c{t2_i}-lr{lr}-decay{decay}-gau{gau_i}-ep{ep}-gr{t1_i}"
                        path = f'./img/{dir}'
                        if not os.path.exists(path):
                            os.mkdir(path)
                        print(dir)
                        ghsom = GHSOM(
                            input_dataset=data,
                            t1=t1_i,
                            t2=t2_i,
                            learning_rate=lr,
                            decay=decay,
                            gaussian_sigma=gau_i,
                        )

                        print("Training...")
                        zero_unit = ghsom.train(
                            epochs_number=ep,
                            dataset_percentage=0.50,
                            min_dataset_size=30,
                            seed=0,
                            grow_maxiter=gr_i,
                        )

                        t = time.time() - start

                        print(f"Elapsed Time:{t}s")

                        # 平均と標準偏差
                        print(f"(誤差平均, 誤差分散):{mean_data_centroid_activation(zero_unit, data)}")
                        print(f"ニューロン使用率:{dispersion_rate(zero_unit, data)}")
                        print(f"マップの数:{number_of_maps(zero_unit)}")
                        print(f"ニューロンの数:{number_of_neurons(zero_unit)}")
                        print("\n")
                        # print(zero_unit)
                        # interactive_plot(zero_unit.child_map)
                        # interactive_plot_with_labels(zero_unit.child_map, data, labels)
                        image_plot_with_labels_save(zero_unit.child_map, data, labels, f"img/{dir}")
                        # plt.show()
                        del ghsom, zero_unit
                        gc.collect()

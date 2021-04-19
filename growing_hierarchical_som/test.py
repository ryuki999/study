"""
test.py
test用のプログラム
"""

import sys
import itertools
import time
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from ghsom_describe import mean_data_centroid_activation, dispersion_rate
from color_plot import interactive_plot_with_labels
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
    # digisデータ
    digits = load_digits()
    # data = digits.data
    # n_samples, n_features = data.shape
    # n_digits = len(np.unique(digits.target))
    # labels = digits.target

    # sample_genomeデータ
    folder = sys.argv[1]
    df0 = read_data(f"{folder}/Bacteria.frq")
    df0["label"] = 0
    df1 = read_data(f"{folder}/Eukaryote.frq")
    df1["label"] = 1
    df2 = read_data(f"{folder}/Virus.frq")
    df2["label"] = 2
    df_concat = pd.concat([df0, df1, df2])

    labels = np.array(df_concat["label"])
    data = np.array(df_concat.drop(columns=["label"]))
    n_samples, n_features = data.shape
    n_digits = len(np.unique(labels))

    # Sars-Cov2データ
    # input_data = sys.argv[1]
    # model_output = sys.argv[2]
    # print(f"input_data:{input_data}")
    # print(f"model_output:{model_output}")
    # header, feature = fasta_to_df(input_data, N, ALL_DATA_HEADER_COLUMNS, N_BASE
    # )
    # df = all_data_df_to_arange_df(header, feature)
    # df1 = df[df["clade"] != "O"]
    # df1["clade"] = df1["clade"].map(
    #     {"G": 0, "GH": 1, "GR": 2, "GV": 3, "L": 4, "S": 5, "V": 6}
    # )
    # labels = np.array(df1["clade"])
    # data = np.array(df1[N_BASE])
    # n_samples, n_features = data.shape
    # n_digits = len(np.unique(labels))

    start = time.time()

    print(type(data))
    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))
    ghsom = GHSOM(
        input_dataset=data,
        t1=0.1,
        t2=0.0001,
        learning_rate=0.15,
        decay=0.95,
        gaussian_sigma=3,
    )

    print("Training...")
    zero_unit = ghsom.train(
        epochs_number=15,
        seed=0,
        min_dataset_size=10,
        grow_maxiter=15,
    )

    # f = open('ghsom.pkl','wb')
    # pickle.dump(zero_unit,f)
    # f = open('ghsom.pkl','rb')
    # zerp_unit = pickle.load(f)
    # f.close

    # print(zero_unit)
    # 平均と標準偏差
    # print(f"(誤差平均, 誤差分散):{mean_data_centroid_activation(zero_unit, data)}")
    # print(f"ニューロン使用率:{dispersion_rate(zero_unit, data)}")
    # interactive_plot(zero_unit.child_map)
    interactive_plot_with_labels(zero_unit.child_map, data, labels)
    plt.show()

"""
test.py
SARS-CoV-2のL,S型に限定してGHSOMを学習させるプログラム
python l_s_ghsom.py input_data model_output_dir

ex)
python l_s_ghsom.py /mnt/mount_point/furukawa_data/To_Furukawa_210318/all_data_odd_penta /mnt/mount_point/penta/

args:
    input_data : 入力データのファイル
    model_output_dir : 画像ファイルとモデル(.pkl)を保存するフォルダ

"""

from sklearn.datasets import load_digits
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
            sample_genome_array.append(one_data)
        sample_genome_df = pd.DataFrame(sample_genome_array)
    f.close()
    return sample_genome_df


if __name__ == "__main__":
    # Sars-Cov2データ
    input_data = sys.argv[1]
    model_output_dir = sys.argv[2]
    print(f"input_data:{input_data}")
    print(f"model_output:{model_output_dir}")
    header, feature = fasta_to_df(input_data, N, ALL_DATA_HEADER_COLUMNS, N_BASE
    )
    df = all_data_df_to_arange_df(header, feature)
    # df1 = df[(df["clade"] == "L") | (df["clade"] == "S")]
    # df1["clade_num"] = df1["clade"].map(
    #     {"L": 0, "S": 1}
    # )
    df1 = df[df["clade"] == "S"]
    df1["clade_num"] = df1["clade"].map(
        {"S": 0}
    )
    labels = np.array(df1["clade_num"])
    data = np.array(df1[N_BASE])
    head = df1.drop(columns=N_BASE).values
    head_columns = df1.drop(columns=N_BASE).columns
    n_samples, n_features = data.shape
    n_digits = len(np.unique(labels))

    start = time.time()
    
    print(df1)
    print(data)
    print(labels)
    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))

    t1 = [0.1]
    t2 = [0.001, 0.0001]
    gaussian_sigma = [3]
    grow_maxiter = [20]
    epochs = [15]
    lr = 0.15
    decay = 0.95

    for t1_i in t1:
        for t2_i in t2:
            for gau_i in gaussian_sigma:
                for ep in epochs:
                    for gr_i in grow_maxiter:
                        start = time.time()
                        dir = f"t1c{t1_i}-t2c{t2_i}-lr{lr}-decay{decay}-gau{gau_i}-ep{ep}-gr{gr_i}"
                        path = f'{model_output_dir}/{dir}'
                        if not os.path.exists(path):
                            os.mkdir(path)
                        if not os.path.exists(f"{model_output_dir}/{dir}.pkl"):
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
                            f = open(f"{model_output_dir}/{dir}.pkl",'wb')
                            pickle.dump(zero_unit,f)
                            f.close
                        f = open(f"{model_output_dir}/{dir}.pkl",'rb')
                        zero_unit = pickle.load(f)
                        f.close

                        with open(f"{model_output_dir}/{dir}.out", "w+") as f:
                            f.write(f"input_data:{input_data}\n")
                            f.write(f"model_output:{model_output_dir}\n")
                            f.write("dataset length: {}\n".format(n_samples))
                            f.write("features per example: {}\n".format(n_features))
                            f.write("number of digits: {}\n".format(n_digits))
                            f.write("id, t1, t2, lr, decay, gau, ep, gr, Elapsed Time, (誤差平均, 誤差分散), ニューロン使用率, 階層数, マップ数, ニューロン数, 黒点割合\n")
                            # data_property, black_point_num = image_plot_with_labels_save(zero_unit.child_map, data, labels, head)
                            data_property, black_point_num = image_plot_with_labels_save(zero_unit.child_map, data, labels, head, path)                 
                            t = time.time() - start
                            f.write(f"{dir}, {t1_i}, {t2_i}, {lr}, {decay}, {gau_i}, {ep}, {gr_i},")
                            # Elapsed Time[s]
                            f.write(f"{np.round(t,4)},")
                            # (誤差平均, 誤差分散)
                            f.write(f"{mean_data_centroid_activation(zero_unit, data)},")
                            # ニューロン使用率
                            f.write(f"{np.round(dispersion_rate(zero_unit, data),4)},")
                            # 階層数
                            num_layers = number_of_layes(zero_unit)
                            f.write(f"{num_layers},")
                            # マップ数
                            f.write(f"{number_of_maps(zero_unit)},")
                            # ニューロン数
                            f.write(f"{number_of_neurons(zero_unit)},")
                            # 黒点割合
                            f.write(f"{np.round(black_point_num / number_of_neurons(zero_unit), 4)}")
                            f.write("\n")

                            # print("head id continent country city host clade_head date")
                            f.write(" ".join([str(i) for i in head_columns]) + f" layer {' '.join([str(i) for i in range(num_layers)])} (y,x)\n")
                            for i,v in data_property.items():
                                f.write(f"{i} {v[-1].split(' ')[0]} ")
                                for vi in range(num_layers):
                                    if vi < len(v):
                                        f.write(f"{v[vi].split(' ')[1]} ")
                                    else:
                                        f.write("None")
                                f.write(f"{v[-1].split(' ')[2]}\n")
                        f.close()
                        del zero_unit
                        gc.collect()
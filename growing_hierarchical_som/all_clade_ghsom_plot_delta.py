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
from color_plot import interactive_plot_with_labels, image_plot
import gc
from sklearn.preprocessing import LabelEncoder
from GHSOM import GHSOM

# モジュール検索パスに，ひとつ上の階層の絶対パスを追加
sys.path.append("..")

from create_dataset.fasta_to_df import all_data_df_to_arange_df, fasta_to_df


if __name__ == "__main__":
    # Sars-Cov2データ
    # model.pklまでのパス
    model = sys.argv[1]
    # 入力データ
    input_data = sys.argv[2]
    output_file = sys.argv[3]
    print(f"model:{model}")
    print(f"input_data:{input_data}")
    print(f"output_file:{output_file}")
    with open(f"{input_data}", encoding="utf-8") as f:
        data = f.read().split("\n")
        header = [data[i] for i in range(0, len(data) - 1, 3)]
        data = [np.array([float(j) for j in data[i].split()]) for i in range(2, len(data), 3)]

    data = np.array(data)
    header = np.array(header).reshape(-1,1)
    n_samples, n_features = data.shape

    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))

    f = open(model,'rb')
    zero_unit = pickle.load(f)
    f.close
    
    with open(f"{output_file}", "w+") as f:
        f.write(f"model:{model}\n")
        f.write(f"input_data:{input_data}\n")
        f.write(f"output_dir:{output_file}\n")
        f.write("dataset length: {}\n".format(n_samples))
        f.write("features per example: {}\n".format(n_features))
        f.write("(誤差平均, 誤差分散), ニューロン使用率, 階層数, マップ数, ニューロン数\n")
        data_property = image_plot(zero_unit.child_map, data, header, data_property={})
        # data_property, black_point_num = image_plot_with_labels_save(zero_unit.child_map, data, labels, head, path, data_property={})                 
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
        f.write(f"{number_of_neurons(zero_unit)}\n")

        # print("head id continent country city host clade_head date")
        f.write("header" + f" layer {' '.join([str(i) for i in range(num_layers)])} (y,x)\n")
        for i,v in data_property.items():
            f.write(f"{i} {v[-1].split(' ')[0]} ")
            for vi in range(num_layers):
                if vi < len(v):
                    f.write(f"{v[vi].split(' ')[1]} ")
                else:
                    f.write("None ")
            f.write(f"{v[-1].split(' ')[2]}\n")
    f.close()
    del zero_unit, data_property
    gc.collect()

import itertools
import pathlib
import pickle
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from tqdm import tqdm
from GHSOM import GHSOM

# ひとつ上の階層の絶対パスを取得
parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())

# モジュール検索パスに，ひとつ上の階層の絶対パスを追加
sys.path.append("..")

from create_dataset.fasta_to_df import all_data_df_to_arange_df, fasta_to_df

N = 3
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

color = [
    "bright red",
    "pure orange",
    "dark violet",
    "very soft orange",
    "lime",
    "blue",
    "fuchsia",
]
rgb = {
    "bright red": [229, 43, 80],
    "pure orange": [255, 191, 0],
    "dark violet": [75, 0, 130],
    "very soft orange": [251, 206, 177],
    # 保留
    "lime": [0, 255, 0],
    "blue": [0, 0, 255],
    "fuchsia": [255, 0, 255],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
}

data_shape=12

def read_data(filename):
    with open(filename) as f:
        data = f.read().split("\n")
        df = []
        for i in range(2, len(data), 3):
            one_data = [int(float(n)) for n in data[i].split()]
            while len(one_data) != 144:
                one_data.append(0)
            df.append(one_data)
        df = pd.DataFrame(df)
    f.close()
    return df


def __gmap_to_matrix(gmap, dataset, labels):
    g = gmap
    gmap = gmap.weights_map[0]
    # マップの横幅
    map_row = data_shape * gmap.shape[0]
    # マップの縦幅
    map_col = data_shape * gmap.shape[1]
    # 各要素に表示させる画像の初期化
    _image = np.empty(shape=(map_row, map_col, 3), dtype=np.int32)
    mapping = [[list() for _ in range(gmap.shape[1])] for _ in range(gmap.shape[0])]
    for idx, label in enumerate(labels):
        winner_neuron = g.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        mapping[r][c].append(label)

    for i in range(0, map_row, data_shape):
        for j in range(0, map_col, data_shape):
            map_label = list(set(mapping[i // data_shape][j // data_shape]))
            if len(map_label) > 1:
                _img = np.full((data_shape, data_shape, 3), rgb["black"])
            elif len(map_label) == 0:
                _img = np.full((data_shape, data_shape, 3), rgb["white"])
            else:
                _img = np.full((data_shape, data_shape, 3), rgb[color[map_label[0]]])
            _image[i : (i + data_shape), j : (j + data_shape)] = _img
    return _image


def __plot_child_with_labels(e, gmap, level, data, labels, associations):
    # マウスクリックの範囲がaxes内の時
    if e.inaxes is not None:
        # coords:座標(0,0)など
        # 2x3のdata_shape8のときe.ydata(0~16),e.xdata(0~32)
        coords = (int(e.ydata // data_shape), int(e.xdata // data_shape))
        # 指定座標のみ抽出
        neuron = gmap.neurons[coords]
        # print(np.array(associations).shape)
        if neuron.child_map is not None:
            # 指定した座標以下のデータとラベルのindexを取得
            assc = associations[coords[0]][coords[1]]
            interactive_plot_with_labels(
                neuron.child_map,
                dataset=data[assc],
                labels=labels[assc],
                num=str(coords),
                level=level + 1,
            )


def interactive_plot_with_labels(gmap, dataset, labels, num="root", level=1):
    """GHSOMのマップをラベルとインタラクティブなプロット

    Args:
    gmap (Neuron): Neuronオブジェクト
    dataset(numpy): 現在のmap以下のdataset
    laeles: 現在のmap以下のlabel
    num: 座標
    level: 階層レベル

    TODO:
    __gmap_to_matrixにその階層のlabelのデータも渡して、そのラベルごとの色で表示させる
    """
    colors = [
        "#E52B50",
        "#FFBF00",
        "#4B0082",
        "#FBCEB1",
        "#7FFFD4",
        "#007FFF",
        "#00FF00",
        "#9966CC",
        "#CD7F32",
        "#89CFF0",
    ]

    sizes = np.arange(0, 60, 6) + 0.5

    # mapサイズ[1]:縦,[0]:横のリスト(ラベルを格納)
    mapping = [
        [list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])
    ]

    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    # 一階層分の参照ベクトルをプロット gmap.weights_map:(tate, yoko, weights)
    ax.imshow(
        __gmap_to_matrix(gmap, dataset, labels),
        # ax.imshow(__gmap_to_matrix(gmap.weights_map, dataset, labels),
        cmap="bone_r",
        interpolation="sinc",
    )
    # マウスをクリックしたとき関数を実行
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: __plot_child_with_labels(
            event, gmap, level, dataset, labels, mapping
        ),
    )
    plt.axis("off")

    for idx, label in tqdm(enumerate(labels)):
        winner_neuron = gmap.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        mapping[r][c].append(idx)

        ax.plot(
            c * data_shape + data_shape / 2,
            r * data_shape + data_shape / 2,
            "o",
            markerfacecolor="None",
            markeredgecolor=colors[label],
            markersize=sizes[label],
            markeredgewidth=0.5,
            label=label,
        )
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(legend_labels, legend_handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
        borderaxespad=0.0,
        mode="expand",
        labelspacing=int((gmap.map_shape()[0] / 9) * data_shape),
    )
    fig.show()


def mean_data_centroid_activation(ghsom, dataset):
    """
    データとGHSOMのニューロンの総誤差の平均と標準偏差を返す
    """
    distances = list()

    for data in dataset:
        _neuron = ghsom
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]
        distances.append(_neuron.activation(data))

    distances = np.asarray(a=distances, dtype=np.float32)
    return distances.mean(), distances.std()


def __number_of_neurons(root):
    """
    GHSOMのneurons数を返す
    """
    r, c = root.child_map.weights_map[0].shape[0:2]
    total_neurons = r * c
    for neuron in root.child_map.neurons.values():
        if neuron.child_map is not None:
            total_neurons += __number_of_neurons(neuron)
    return total_neurons


def dispersion_rate(ghsom, dataset):
    """
    GHSOMの全ニューロンの中で使用されているニューロンの割合を返す
    """
    used_neurons = dict()
    for data in dataset:
        gsom_reference = ""
        neuron_reference = ""
        _neuron = ghsom
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]

            gsom_reference = str(_gsom)
            neuron_reference = str(_neuron)

        used_neurons[
            "{}-{}-{}".format(gsom_reference, neuron_reference, _neuron.position)
        ] = True
    used_neurons = len(used_neurons)

    return __number_of_neurons(ghsom) / used_neurons


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

    print(f"input_data:{plot_data}")
    print(f"model_output:{model_path}")
    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))

    f = open(model_path, "rb")
    zero_unit = pickle.load(f)
    f.close

    # print(zero_unit)
    # 平均と標準偏差
    print(f"(誤差平均, 誤差分散):{mean_data_centroid_activation(zero_unit, data)}")
    print(f"ニューロン使用率:{dispersion_rate(zero_unit, data)}")
    # interactive_plot(zero_unit.child_map)
    interactive_plot_with_labels(zero_unit.child_map, data, labels)
    plt.show()

"""
image_plot.py
特徴量をそのまま画像形式でプロットするための関数をまとめたモジュール
"""

from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
DATA_SHAPE = 12

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


def __gmap_to_matrix(gmap):
    gmap = gmap[0]
    # マップの横幅
    map_row = DATA_SHAPE * gmap.shape[0]
    # マップの縦幅
    map_col = DATA_SHAPE * gmap.shape[1]
    # 各要素に表示させる画像の初期化
    _image = np.empty(shape=(map_row, map_col), dtype=np.float32)
    for i in range(0, map_row, DATA_SHAPE):
        for j in range(0, map_col, DATA_SHAPE):
            # 各要素に表示させるニューロンの取得
            neuron = gmap[i // DATA_SHAPE, j // DATA_SHAPE]
            # ニューロンを変形させて画像として表示
            _image[i : (i + DATA_SHAPE), j : (j + DATA_SHAPE)] = np.reshape(
                neuron, newshape=(DATA_SHAPE, DATA_SHAPE)
            )
    return _image


def __plot_child(e, gmap, level):
    if e.inaxes is not None:
        coords = (int(e.ydata // DATA_SHAPE), int(e.xdata // DATA_SHAPE))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_plot(neuron.child_map, num=str(coords), level=level + 1)


def interactive_plot(gmap, num="root", level=1):
    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap="bone_r", interpolation="sinc")
    fig.canvas.mpl_connect(
        "button_press_event", lambda event: __plot_child(event, gmap, level)
    )
    plt.axis("off")
    fig.show()


def __plot_child_with_labels(e, gmap, level, data, labels, associations):
    """
    gsom.neurons:
    {(0,0):
    position (0, 0) -- map dimensions (3, 3, 64) -- input dataset 197 element(s) -- level 0
        position (0, 0) -- map dimensions (4, 2, 64) -- input dataset 40 element(s) -- level 1
            position (0, 0) -- map dimensions (5, 2, 64) -- input dataset 1 element(s) -- level 2
                position (0, 0) -- map dimensions (3, 2, 64) -- input dataset 1 element(s) -- level 3
                position (0, 1) -- map dimensions (3, 2, 64) -- input dataset 0 element(s) -- level 3
                position (2, 0) -- map dimensions (3, 2, 64) -- input dataset 0 element(s) -- level 3
                position (2, 1) -- map dimensions (3, 2, 64) -- input dataset 0 element(s) -- level 3
                position (1, 0) -- map dimensions (3, 2, 64) -- input dataset 0 element(s) -- level 3
                position (1, 1) -- map dimensions (3, 2, 64) -- input dataset 0 element(s) -- level 3
            position (0, 1) -- map dimensions (5, 2, 64) -- input dataset 7 element(s) -- level 2
    """
    # マウスクリックの範囲がaxes内の時
    if e.inaxes is not None:
        # coords:座標(0,0)など
        # 2x3のDATA_SHAPE8のときe.ydata(0~16),e.xdata(0~32)
        coords = (int(e.ydata // DATA_SHAPE), int(e.xdata // DATA_SHAPE))
        # 指定座標のみ抽出
        neuron = gmap.neurons[coords]
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
    """

    sizes = np.arange(0, 60, 6) + 0.5

    # mapサイズ[1]:縦,[0]:横のリスト(ラベルを格納)
    mapping = [
        [list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])
    ]

    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    # 一階層分の参照ベクトルをプロット gmap.weights_map:(tate, yoko, weights)
    ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap="bone_r", interpolation="sinc")
    # マウスをクリックしたとき関数を実行
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: __plot_child_with_labels(
            event, gmap, level, dataset, labels, mapping
        ),
    )
    plt.axis("off")

    for idx, label in enumerate(labels):
        winner_neuron = gmap.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        mapping[r][c].append(idx)

        ax.plot(
            c * DATA_SHAPE + DATA_SHAPE / 2,
            r * DATA_SHAPE + DATA_SHAPE / 2,
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
        labelspacing=int((gmap.map_shape()[0] / 9) * DATA_SHAPE),
    )
    fig.show()

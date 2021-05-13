from collections import OrderedDict, defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import gc

DATA_SHAPE = 12

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


def __gmap_to_matrix(gmap, dataset, labels):
    g = gmap
    gmap = gmap.weights_map[0]
    # マップの横幅
    map_row = DATA_SHAPE * gmap.shape[0]
    # マップの縦幅
    map_col = DATA_SHAPE * gmap.shape[1]
    # 各要素に表示させる画像の初期化
    _image = np.empty(shape=(map_row, map_col, 3), dtype=np.int32)
    mapping = [[list() for _ in range(gmap.shape[1])] for _ in range(gmap.shape[0])]
    for idx, label in enumerate(labels):
        winner_neuron = g.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        mapping[r][c].append(label)

    for i in range(0, map_row, DATA_SHAPE):
        for j in range(0, map_col, DATA_SHAPE):
            map_label = list(set(mapping[i // DATA_SHAPE][j // DATA_SHAPE]))
            if len(map_label) > 1:
                c = rgb["black"]
            elif len(map_label) == 0:
                c = rgb["white"]
            else:
                c = rgb[color[map_label[0]]]
            white_vary = np.full((DATA_SHAPE, DATA_SHAPE, 3), rgb["white"])
            _image[i : (i + DATA_SHAPE), j : (j + DATA_SHAPE)] = white_vary

            _img = np.full((DATA_SHAPE-2, DATA_SHAPE-2, 3), c)
            _image[i+1 : (i + DATA_SHAPE)-1, j+1 : (j + DATA_SHAPE)-1] = _img
    return _image


def __plot_child_with_labels(e, gmap, level, data, labels, associations):
    # マウスクリックの範囲がaxes内の時
    if e.inaxes is not None:
        # coords:座標(0,0)など
        # 2x3のDATA_SHAPE8のときe.ydata(0~16),e.xdata(0~32)
        coords = (int(e.ydata // DATA_SHAPE), int(e.xdata // DATA_SHAPE))
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
    """

    sizes = np.arange(0, 60, 6) + 0.5

    # mapサイズ[1]:縦,[0]:横のリスト(ラベルを格納)
    mapping = [[list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])]

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
        lambda event: __plot_child_with_labels(event, gmap, level, dataset, labels, mapping),
    )
    plt.axis("off")

    for idx, label in tqdm(enumerate(labels)):
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

def plot(args):

    img, save_dir, _num = args
    fig, ax = plt.subplots(num=_num)
    
    # 一階層分の参照ベクトルをプロット gmap.weights_map:(tate, yoko, weights)
    ax.imshow(
        img,
        cmap="bone_r",
        interpolation="sinc",
    )

    plt.axis("off")
    plt.savefig(f"{save_dir}/{_num}.png")
    del img, save_dir, _num
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    gc.collect()

def image_plot_with_labels_save(gmap, dataset, labels, head, save_dir=None, data_property={}, parent_map="root", num="root", level=1):
    """GHSOMのマップをラベルとインタラクティブなプロット

    Args:
    gmap (Neuron): Neuronオブジェクト
    dataset(numpy): 現在のmap以下のdataset
    laeles: 現在のmap以下のlabel
    save_dir: pngを保存するディレクトリ
    parent_map: 親の階層のmap
    num: 座標
    level: 階層レベル
    """
    # mapサイズ[1]:縦,[0]:横のリスト(ラベルを格納)
    mapping = [[list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])]
    plot_labels = [[list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])]

    for idx, label in enumerate(labels):
        winner_neuron = gmap.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        mapping[r][c].append(idx)
        plot_labels[r][c].append(label)
        key = " ".join([str(i) for i in head[idx]])
        if key not in data_property:
            data_property[key] = []
        data_property[key].append(f"{level} {parent_map} ({r},{c})")
        # print(" ".join(head.iloc[idx,:].values), f"階層{level} 親マップ{parent_map} 座標{r} {c}")
    
    _num = f"level {level} --parent map {parent_map} --num of data {len(labels)}"

    black_point_num = 0
    for r in range(np.array(plot_labels).shape[0]):
        for c in range(np.array(plot_labels).shape[1]):
            # print(plot_label[r][c])
            if len(list(set(plot_labels[r][c]))) != 1:
                black_point_num += 1

    if save_dir:
        p = Pool(1)
        p.map(plot, [[__gmap_to_matrix(gmap, dataset, labels), save_dir, _num]])
        p.close()

    for i in range(gmap.map_shape()[0]):
        for j in range(gmap.map_shape()[1]):
            coords = (i,j)
            # print(parent_map, coords, len(mapping[i][j]))
            # 指定座標のみ抽出
            neuron = gmap.neurons[coords]
            current_map = parent_map + f"({i},{j})"
            if neuron.child_map is not None:
                # 指定した座標以下のデータとラベルのindexを取得
                assc = mapping[coords[0]][coords[1]]
                data_property, _black_point_num = image_plot_with_labels_save(
                    neuron.child_map,
                    dataset=dataset[assc],
                    labels=labels[assc],
                    head=head[assc],
                    save_dir=save_dir,
                    data_property=data_property,
                    parent_map=current_map,
                    num=str(coords),
                    level=level + 1,
                )

                black_point_num += _black_point_num

    del gmap, dataset, labels, mapping, coords, neuron, current_map
    gc.collect() 

    return data_property, black_point_num
"""
ghsom_describe.py
GHSOMの特徴を示すための関数をまとめたモジュール
"""

import numpy as np


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

def number_of_maps(root):
    """
    GHSOMのmap数を返す
    """
    total_maps = 1
    for neuron in root.child_map.neurons.values():
        if neuron.child_map is not None:
            total_maps += number_of_maps(neuron)
    return total_maps

def number_of_neurons(root):
    return __number_of_neurons(root)

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

        used_neurons["{}-{}-{}".format(gsom_reference, neuron_reference, _neuron.position)] = True
    used_neurons = len(used_neurons)

    return __number_of_neurons(ghsom) / used_neurons

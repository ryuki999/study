"""
parallel_create_odd_ratio.py
連続塩基頻度のオッズ比の計算用プログラム

並列計算用に全ての塩基配列を一つの配列にまとめてからオッズ比の計算を行い、ファイルに書き込む

Args:
    input_filename = sys.argv[1] : 塩基配列が書かれたfastaファイル
    output_filename = sys.argv[2] : 出力用のファイル

    
・130,750件 / 1件当たり30kの塩基配列
・実行環境
-研究室サーバ133.35.159.13

・5連続オッズ比生成の実行時間
-Elapsed time:176.0197s(3分)

・8連続オッズ比生成の実行時間
-Elapsed time:1099.674s(18分)
-ファイルサイズ47Gくらいあった:+1
"""

import itertools
import multiprocessing
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas
from tqdm import tqdm

# from penta_continuous import N_BASE

N = 5
BASE = ["A", "T", "G", "C"]
N_BASE = ["".join(i) for i in list(itertools.product(BASE, repeat=N))]


def seq_to_odd(line):
    """塩基配列からオッズ比を計算しヘッダーと結合して一要素とする
    Args:
        line (str): >>header\nATCGATGCA...
    Return:
        (str): >>header\n 0.9821 0.8912 1.3121 0.9812 ... \n
    """
    header, seq = line.split("\n")
    n_continuous_base_odd = calculate_odd_ratio(seq)
    return (
        header + "\n" + header + "\n" + " ".join([str(n_continuous_base_odd[b]) for b in N_BASE]) + "\n"
    )

def calculate_odd_ratio(seq):
    """オッズ比を計算する
    Args:
        seq (list): 塩基配列 ex)ATGCGCTATAA
    Return:
        n_continuous_base_odd (dict): オッズ比を格納した辞書
        
        ex) {"AAAAA":0.9842, "AAAAT":0.99332,...,"CCCCC":0.8932}
    """
    n_continuous_base_odd = defaultdict(float)
    one_base = defaultdict(int)
    n_base = defaultdict(int)
    # 全ての塩基配列を一塩基ずつ読み込み
    for i in range(len(seq)):
        if seq[i] in BASE:
            # 1塩基の頻度計算
            one_base[seq[i]] += 1
        # 連続塩基の頻度計算
        if i + N < len(seq):
            flag = 0
            for j in seq[i:i+N]:
                if j not in BASE:
                    flag = 1
            if flag != 1:
                n_base[seq[i : i + N]] += 1
    total_n_continuous_base_freq = np.sum(list(n_base.values()))
    total_one_base_freq = np.sum(list(one_base.values()))

    # オッズ比の計算
    for key, val in n_base.items():
        one_base_freq = 1
        for k in key:
            one_base_freq *= one_base[k] / total_one_base_freq

        n_base_freq = val / total_n_continuous_base_freq
        n_continuous_base_odd[key] = "{:.6f}".format(n_base_freq / one_base_freq)
    return n_continuous_base_odd

def parallel_calculate(seq_array):
    """オッズ比を並列計算する
    Args:
        seq_array (list): ["header\nATGCGCTAA..", "header\nATGCGATAA..",...,"header\nATGCGCTAA.."]
    Return:
        seq_array (list): [0.9842, 0.99332, ..., 0.8932]
                        左から["AAAAA", "AAAAT",... ,"CCCCC"]
    """
    p = Pool(multiprocessing.cpu_count() - 1)
    seq_array = p.map(seq_to_odd, tqdm(seq_array))
    p.close()

    return seq_array

def main(filename, output_filename, number_of_simultaneous_calculations):
    seq_array = []
    seq = None

    print(f"create {N} continuous base odd ratio")
    print(f"input_data:{filename}")
    print(f"output_filename:{output_filename}")
    start = time.time()
   
    with open(filename, "r", encoding="utf-8") as r:
        with open(output_filename, "w", encoding="utf-8") as w:
            for line in tqdm(r.readlines()):
                if ">" in line:
                    if seq is not None:
                        seq_array.append(seq)
                    seq = line
                else:
                    seq += line.strip()
                # 同時計算数の上限に達したとき計算・出力
                if len(seq_array) == int(number_of_simultaneous_calculations):
                    seq_array = parallel_calculate(seq_array)
                    for line in seq_array:
                        w.write(line)
                    seq_array = []

            # 端数部分の配列を計算・出力
            seq_array.append(seq)
            seq_array = parallel_calculate(seq_array)
            for line in seq_array:
                w.write(line)
    t = time.time() - start
    print(f"Elapsed time:{t} s")

if __name__ in "__main__":
    filename = sys.argv[1]
    output_filename = sys.argv[2]
    # 同時に計算する配列数
    number_of_simultaneous_calculations = sys.argv[3]
    
    main(filename, output_filename, number_of_simultaneous_calculations)

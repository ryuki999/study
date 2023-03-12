"""train_ghsom.py
任意のゲノムデータに対して、GHSOMの学習を行うプログラム
python train_ghsom.py input_data model_output_dir

ex)
python train_ghsom.py ~/furukawa/furukawa_data/To_Furukawa_210318/all_data_odd_penta ~/penta/

args:
    input_data : 入力データのファイル
    model_output_dir : モデル(.pkl)を保存するフォルダ
"""

import os
import pickle
import sys
import time
import numpy as np
import gc
from GHSOM import GHSOM

if __name__ == "__main__":
    # Sars-Cov2データ
    input_data = sys.argv[1]
    model_output_dir = sys.argv[2]
    print(f"input_data:{input_data}")
    print(f"model_output:{model_output_dir}")
    with open(f"{input_data}", encoding="utf-8") as f:
        data = f.read().split("\n")
        header = [data[i] for i in range(0, len(data) - 1, 3)]
        data = [np.array([float(j) for j in data[i].split()]) for i in range(2, len(data), 3)]


    data = np.array(data)
    header = np.array(header).reshape(-1,1)
    n_samples, n_features = data.shape

    start = time.time()
    
    # print(df)
    print(data)
    # print(labels)
    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    # print("number of digits: {}\n".format(n_digits))

    lr = 0.15
    decay = 0.95

    t1 = [0.1] # [0.1, 0.01, 0.001]
    t2 = [0.01] # [0.001, 0.0001]
    gaussian_sigma = [3]
    grow_maxiter = [10]
    epochs = [10]
    
    for t1_i in t1:
        for t2_i in t2:
            for gau_i in gaussian_sigma:
                for ep in epochs:
                    for gr_i in grow_maxiter:
                        start = time.time()
                        dir = f"t1c{t1_i}-t2c{t2_i}-lr{lr}-decay{decay}-gau{gau_i}-ep{ep}-gr{gr_i}"
                        print(dir)
                        path = f'{model_output_dir}/{dir}'
                        #if not os.path.exists(path):
                        #    os.mkdir(path)
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
                        else:
                            print("Already Exists\n")
                            f = open(f"{model_output_dir}/{dir}.pkl",'rb')
                            zero_unit = pickle.load(f)
                            f.close
                        t = time.time() - start
                        # Elapsed Time[s]
                        print(f"{np.round(t,4)},")

                        del zero_unit
                        gc.collect()

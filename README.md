# GHSOMおよびデータセット作成に関するリポジトリ

## growing_hierarchical_som
成長する階層型自己組織化マップ(Growing Hierarchical Self-Organizing Map)の開発用フォルダ

GHSOMの実装は以下のリポジトリのものを流用しており、これを可視化するプログラムを主に開発した
Github: https://github.com/enricivi/growing_hierarchical_som


### ファイル構成
```
.
├── GHSOM.py
├── GSOM.py
├── color_plot.py
├── ghsom_describe.py # GHSOMの統計量計算間数群
├── model
│   └── t1c0.1-t2c0.01-lr0.15-decay0.95-gau3-ep10-gr10.pkl
├── neuron
│   ├── __init__.py
│   ├── neuron.py
│   └── neuron_builder.py
├── output
│   └── out1.txt
├── plot_ghsom.py # 唯一のパラメータで任意(今回は新型コロナ)のデータに対してGHSOM解析
├── sample_data # サンプル入力データ
│   ├── Bacteria.frq
│   ├── Eukaryote.frq
│   ├── Virus.frq
│   └── input.dat
└── train_ghsom.py # 複数のパラメータで任意(今回は新型コロナ)のデータに対してGHSOM解析
```

### 使用環境構築
実行環境は元のGHSOMの開発環境と同じく、以下が必要となる.
```
Python 3.6.5
progressbar2==3.37.1
```

### 使用方法
以下の順に実行すると、GHSOMの可視化結果と、入力データのGHSOMへのPLOT結果が出力される.
```
python train_ghsom.py sample_data/input.dat model
python plot_ghsom.py model/t1c0.1-t2c0.01-lr0.15-decay0.95-gau3-ep10-gr10.pkl sample_data/input.dat output/out1.txt
```

* 準備するもの
    * input data: sample_dataフォルダ以下のような形式
* 生成物
    * model
    * output/out.txt


## create_dataset
ゲノム配列(メタデータ含む)のデータセット作成に関するファイル群

### ファイル構成
```
.
├── create_odd_ratio.py # 連続塩基頻度のオッズ比を作成するプログラム(シングルコア)
└── parallel_create_odd_ratio.py # 連続塩基頻度のオッズ比を作成するプログラム(マルチコア,並列計算)
```

### 使用方法
各プログラム内で、N(連続塩基数)を指定した後、以下のように引数を指定して実行すると、「入力マルチFASTAファイル名」のゲノムデータから、「出力ファイル名」の名前で連続頻度のオッズ比が生成される.
```
python create_odd_ratio.py 入力マルチFASTAファイル名 出力ファイル名
python parallel_create_odd_ratio.py 入力マルチFASTAファイル名 出力ファイル名 同時に計算する配列数
```

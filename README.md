# GHSOMおよびデータセット作成に関するリポジトリ

## growing_hierarchical_som
成長する階層型自己組織化マップ(Growing Hierarchical Self-Organizing Map)の開発用フォルダ

### ファイル構成
```
.
├── GHSOM.py
├── GSOM.py
├── all_clade_ghsom_plot_delta.py # 唯一のパラメータで任意(今回は新型コロナ)のデータに対してGHSOM解析
├── color_plot.py 
├── ghsom_describe.py # GHSOMの統計量
├── image_plot.py
├── neuron
│   ├── __init__.py
│   ├── neuron.py
│   └── neuron_builder.py
├── param_test.py # 複数のパラメータでサンプルデータに対してGHSOM解析
├── sars2_ghsom.py # 複数のパラメータで任意(今回は新型コロナ)のデータに対してGHSOM解析
```

### 使用方法

```
```


## create_dataset
ゲノム配列(メタデータ含む)のデータセット作成に関するファイル群

### ファイル構成
```
.
├── create_odd_ratio.py # 連続塩基頻度のオッズ比を作成するプログラム(シングルコア)
├── fasta_to_df.py # 塩基頻度データおよびメタデータ読み込みに関する関数群
└── parallel_create_odd_ratio.py # 連続塩基頻度のオッズ比を作成するプログラム(マルチコア,並列計算)
```

### 使用方法
各プログラム内で、N(連続塩基数)を指定した後、以下のように引数を指定して実行すると、「入力マルチFASTAファイル名」のゲノムデータから、「出力ファイル名」の名前で連続頻度のオッズ比が生成される.
```
python create_odd_ratio.py 入力マルチFASTAファイル名 出力ファイル名
python parallel_create_odd_ratio.py 入力マルチFASTAファイル名 出力ファイル名 同時に計算する配列数
```

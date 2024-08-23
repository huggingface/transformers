<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Model training anatomy

モデルトレーニングの効率を向上させるために適用できるパフォーマンス最適化テクニックを理解するには、トレーニング中にGPUがどのように利用されるか、および実行される操作に応じて計算強度がどのように変化するかを理解することが役立ちます。

まずは、GPUの利用例とモデルのトレーニング実行に関する示唆に富む例を探求することから始めましょう。デモンストレーションのために、いくつかのライブラリをインストールする必要があります:

```bash
pip install transformers datasets accelerate nvidia-ml-py3
```

`nvidia-ml-py3` ライブラリは、Python内からモデルのメモリ使用状況をモニターすることを可能にします。おそらく、ターミナルでの `nvidia-smi` コマンドについてはお聞きかもしれませんが、このライブラリを使用すると、Pythonから同じ情報にアクセスできます。

それから、いくつかのダミーデータを作成します。100から30000の間のランダムなトークンIDと、分類器のためのバイナリラベルです。合計で、512のシーケンスがあり、それぞれの長さは512で、PyTorchフォーマットの [`~datasets.Dataset`] に格納されます。


```py
>>> import numpy as np
>>> from datasets import Dataset


>>> seq_len, dataset_size = 512, 512
>>> dummy_data = {
...     "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
...     "labels": np.random.randint(0, 1, (dataset_size)),
... }
>>> ds = Dataset.from_dict(dummy_data)
>>> ds.set_format("pt")
```


[`Trainer`]を使用してGPU利用率とトレーニング実行の要約統計情報を表示するために、2つのヘルパー関数を定義します。


```py
>>> from pynvml import *


>>> def print_gpu_utilization():
...     nvmlInit()
...     handle = nvmlDeviceGetHandleByIndex(0)
...     info = nvmlDeviceGetMemoryInfo(handle)
...     print(f"GPU memory occupied: {info.used//1024**2} MB.")


>>> def print_summary(result):
...     print(f"Time: {result.metrics['train_runtime']:.2f}")
...     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
...     print_gpu_utilization()
```

以下は、無料のGPUメモリから開始していることを確認しましょう：


```py
>>> print_gpu_utilization()
GPU memory occupied: 0 MB.
```

GPUメモリがモデルを読み込む前のように占有されていないように見えます。これがお使いのマシンでの状況でない場合は、GPUメモリを使用しているすべてのプロセスを停止してください。ただし、すべての空きGPUメモリをユーザーが使用できるわけではありません。モデルがGPUに読み込まれると、カーネルも読み込まれ、1〜2GBのメモリを使用することがあります。それがどれくらいかを確認するために、GPUに小さなテンソルを読み込むと、カーネルも読み込まれます。


```py
>>> import torch


>>> torch.ones((1, 1)).to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 1343 MB.
```

カーネルだけで1.3GBのGPUメモリを使用していることがわかります。次に、モデルがどれだけのスペースを使用しているかを見てみましょう。

## Load Model

まず、`bert-large-uncased` モデルを読み込みます。モデルの重みを直接GPUに読み込むことで、重みだけがどれだけのスペースを使用しているかを確認できます。


```py
>>> from transformers import AutoModelForSequenceClassification


>>> model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 2631 MB.
```

モデルの重みだけで、GPUメモリを1.3 GB使用していることがわかります。正確な数値は、使用している具体的なGPUに依存します。新しいGPUでは、モデルの重みが最適化された方法で読み込まれるため、モデルの使用を高速化することがあるため、モデルがより多くのスペースを占有することがあります。さて、`nvidia-smi` CLIと同じ結果が得られるかを簡単に確認することもできます。


```bash
nvidia-smi
```

```bash
Tue Jan 11 08:58:05 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    39W / 300W |   2631MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3721      C   ...nvs/codeparrot/bin/python     2629MiB |
+-----------------------------------------------------------------------------+
```

前回と同じ数値を取得し、16GBのメモリを搭載したV100 GPUを使用していることがわかります。さて、モデルのトレーニングを開始し、GPUメモリの消費がどのように変化するかを確認してみましょう。まず、いくつかの標準的なトレーニング引数を設定します:


```py
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```

<Tip>

複数の実験を実行する予定がある場合、実験間でメモリを適切にクリアするために、実験の間に Python カーネルを再起動してください。

</Tip>

## Memory utilization at vanilla training

[`Trainer`] を使用して、GPU パフォーマンスの最適化テクニックを使用せずにバッチサイズ 4 でモデルをトレーニングしましょう：


```py
>>> from transformers import TrainingArguments, Trainer, logging

>>> logging.set_verbosity_error()


>>> training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
>>> trainer = Trainer(model=model, args=training_args, train_dataset=ds)
>>> result = trainer.train()
>>> print_summary(result)
```

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```

既に、比較的小さいバッチサイズでも、GPUのほとんどのメモリがすでに使用されていることがわかります。しかし、より大きなバッチサイズを使用することは、しばしばモデルの収束が速くなったり、最終的な性能が向上したりすることがあります。したがって、理想的には、バッチサイズをモデルの要件に合わせて調整したいのですが、GPUの制限に合わせて調整する必要はありません。興味深いことに、モデルのサイズよりもはるかに多くのメモリを使用しています。なぜそうなるのかを少し理解するために、モデルの操作とメモリの必要性を見てみましょう。


## Anatomy of Model's Operations

Transformerアーキテクチャには、計算強度によって以下の3つの主要な操作グループが含まれています。

1. **テンソルの収縮**

    線形層とMulti-Head Attentionのコンポーネントは、すべてバッチ処理された **行列-行列の乗算** を行います。これらの操作は、Transformerのトレーニングにおいて最も計算集約的な部分です。

2. **統計的正規化**

    Softmaxと層正規化は、テンソルの収縮よりも計算負荷が少なく、1つまたは複数の **縮約操作** を含み、その結果がマップを介して適用されます。

3. **要素ごとの演算子**

    これらは残りの演算子です：**バイアス、ドロップアウト、活性化、および残差接続** です。これらは最も計算集約的な操作ではありません。

パフォーマンスのボトルネックを分析する際に、この知識は役立つことがあります。

この要約は、[Data Movement Is All You Need: Optimizing Transformers 2020に関するケーススタディ](https://arxiv.org/abs/2007.00072)から派生しています。

## Anatomy of Model's Memory

モデルのトレーニングがGPUに配置されたモデルよりもはるかに多くのメモリを使用することを見てきました。これは、トレーニング中にGPUメモリを使用する多くのコンポーネントが存在するためです。GPUメモリ上のコンポーネントは以下の通りです：

1. モデルの重み
2. オプティマイザの状態
3. 勾配
4. 勾配計算のために保存された前向き活性化
5. 一時バッファ
6. 機能固有のメモリ

通常、AdamWを使用して混合精度でトレーニングされたモデルは、モデルパラメータごとに18バイトとアクティベーションメモリが必要です。推論ではオプティマイザの状態と勾配は不要ですので、これらを差し引くことができます。したがって、混合精度の推論においては、モデルパラメータごとに6バイトとアクティベーションメモリが必要です。

詳細を見てみましょう。


**モデルの重み:**

- fp32トレーニングのパラメーター数 * 4バイト
- ミックスプレシジョントレーニングのパラメーター数 * 6バイト（メモリ内にfp32とfp16のモデルを維持）

**オプティマイザの状態:**

- 通常のAdamWのパラメーター数 * 8バイト（2つの状態を維持）
- 8-bit AdamWオプティマイザのパラメーター数 * 2バイト（[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)のようなオプティマイザ）
- モーメンタムを持つSGDのようなオプティマイザのパラメーター数 * 4バイト（1つの状態を維持）

**勾配**

- fp32またはミックスプレシジョントレーニングのパラメーター数 * 4バイト（勾配は常にfp32で保持）

**フォワードアクティベーション**

- サイズは多くの要因に依存し、主要な要因はシーケンスの長さ、隠れ層のサイズ、およびバッチサイズです。

フォワードとバックワードの関数によって渡され、返される入力と出力、および勾配計算のために保存されるフォワードアクティベーションがあります。

**一時的なメモリ**

さらに、計算が完了した後に解放されるさまざまな一時変数がありますが、これらは一時的に追加のメモリを必要とし、OOMに達する可能性があります。したがって、コーディング時にはこのような一時変数に戦略的に考え、必要なくなったら明示的に解放することが非常に重要です。

**機能固有のメモリ**

次に、ソフトウェアには特別なメモリ要件がある場合があります。たとえば、ビームサーチを使用してテキストを生成する場合、ソフトウェアは複数の入力と出力のコピーを維持する必要があります。

**`forward`と`backward`の実行速度**

畳み込み層と線形層では、バックワードにフォワードと比べて2倍のFLOPSがあり、一般的には約2倍遅くなります（バックワードのサイズが不便であることがあるため、それ以上になることがあります）。 アクティベーションは通常、バンド幅制限されており、バックワードでアクティベーションがフォワードよりも多くのデータを読むことが一般的です（たとえば、アクティベーションフォワードは1回読み取り、1回書き込み、アクティベーションバックワードはフォワードのgradOutputおよび出力を2回読み取り、1回書き込みます）。

ご覧の通り、GPUメモリを節約したり操作を高速化できる可能性のあるいくつかの場所があります。 GPUの利用と計算速度に影響を与える要因を理解したので、パフォーマンス最適化の技術については、[単一GPUでの効率的なトレーニングのための方法とツール](perf_train_gpu_one)のドキュメンテーションページを参照してください。

詳細を見てみましょう。







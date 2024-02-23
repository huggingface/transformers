<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ このファイルはMarkdownですが、Hugging Faceのdoc-builder（MDXに類似）向けの特定の構文を含んでいるため、
Markdownビューアでは正しく表示されないことに注意してください。
-->

# Benchmarks

<Tip warning={true}>

Hugging Faceのベンチマークツールは非推奨であり、Transformerモデルの速度とメモリの複雑さを測定するために外部のベンチマークライブラリを使用することをお勧めします。

</Tip>

[[open-in-colab]]

🤗 Transformersモデルをベンチマークし、ベストプラクティス、すでに利用可能なベンチマークについて見てみましょう。

🤗 Transformersモデルをベンチマークする方法について詳しく説明したノートブックは[こちら](https://github.com/huggingface/notebooks/tree/main/examples/benchmark.ipynb)で利用できます。

## How to benchmark 🤗 Transformers models

[`PyTorchBenchmark`]クラスと[`TensorFlowBenchmark`]クラスを使用すると、🤗 Transformersモデルを柔軟にベンチマークできます。
ベンチマーククラスを使用すると、_ピークメモリ使用量_ および _必要な時間_ を _推論_ および _トレーニング_ の両方について測定できます。

<Tip>

ここでの _推論_ は、単一のフォワードパスによって定義され、 _トレーニング_ は単一のフォワードパスと
バックワードパスによって定義されます。

</Tip>

ベンチマーククラス[`PyTorchBenchmark`]と[`TensorFlowBenchmark`]は、それぞれのベンチマーククラスに対する適切な設定を含む [`PyTorchBenchmarkArguments`] および [`TensorFlowBenchmarkArguments`] タイプのオブジェクトを必要とします。
[`PyTorchBenchmarkArguments`] および [`TensorFlowBenchmarkArguments`] はデータクラスであり、それぞれのベンチマーククラスに対するすべての関連する設定を含んでいます。
次の例では、タイプ _bert-base-cased_ のBERTモデルをベンチマークする方法が示されています。

<frameworkcontent>
<pt>
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

>>> args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
>>> benchmark = PyTorchBenchmark(args)
```
</pt>
<tf>
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

>>> args = TensorFlowBenchmarkArguments(
...     models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> benchmark = TensorFlowBenchmark(args)
```
</tf>
</frameworkcontent>


ここでは、ベンチマーク引数のデータクラスに対して、`models`、`batch_sizes`
および`sequence_lengths`の3つの引数が指定されています。引数`models`は必須で、
[モデルハブ](https://huggingface.co/models)からのモデル識別子の`リスト`を期待し
ます。`batch_sizes`と`sequence_lengths`の2つの`リスト`引数は
モデルのベンチマーク対象となる`input_ids`のサイズを定義します。
ベンチマーク引数データクラスを介して設定できる他の多くのパラメータがあります。これらの詳細については、直接ファイル
`src/transformers/benchmark/benchmark_args_utils.py`、
`src/transformers/benchmark/benchmark_args.py`（PyTorch用）、および`src/transformers/benchmark/benchmark_args_tf.py`（Tensorflow用）
を参照するか、次のシェルコマンドをルートから実行すると、PyTorchとTensorflowのそれぞれに対して設定可能なすべてのパラメータの記述的なリストが表示されます。

<frameworkcontent>
<pt>
```bash
python examples/pytorch/benchmarking/run_benchmark.py --help
```

インスタンス化されたベンチマークオブジェクトは、単に `benchmark.run()` を呼び出すことで実行できます。


```py
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.006     
google-bert/bert-base-uncased          8               32            0.006     
google-bert/bert-base-uncased          8              128            0.018     
google-bert/bert-base-uncased          8              512            0.088     
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1227
google-bert/bert-base-uncased          8               32            1281
google-bert/bert-base-uncased          8              128            1307
google-bert/bert-base-uncased          8              512            1539
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 08:58:43.371351
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</pt>
<tf>
```bash
python examples/tensorflow/benchmarking/run_benchmark_tf.py --help
```

インスタンス化されたベンチマークオブジェクトは、単に `benchmark.run()` を呼び出すことで実行できます。



```py
>>> results = benchmark.run()
>>> print(results)
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.005
google-bert/bert-base-uncased          8               32            0.008
google-bert/bert-base-uncased          8              128            0.022
google-bert/bert-base-uncased          8              512            0.105
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1330
google-bert/bert-base-uncased          8               32            1330
google-bert/bert-base-uncased          8              128            1330
google-bert/bert-base-uncased          8              512            1770
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:26:35.617317
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</tf>
</frameworkcontent>

デフォルトでは、_推論時間_ と _必要なメモリ_ がベンチマークされます。
上記の例の出力では、最初の2つのセクションが _推論時間_ と _推論メモリ_ 
に対応する結果を示しています。さらに、計算環境に関するすべての関連情報、
例えば GPU タイプ、システム、ライブラリのバージョンなどが、_ENVIRONMENT INFORMATION_ の下に表示されます。この情報は、[`PyTorchBenchmarkArguments`] 
および [`TensorFlowBenchmarkArguments`] に引数 `save_to_csv=True` 
を追加することで、オプションで _.csv_ ファイルに保存することができます。この場合、各セクションは別々の _.csv_ ファイルに保存されます。_.csv_ 
ファイルへのパスは、データクラスの引数を使用してオプションで定義できます。

モデル識別子、例えば `google-bert/bert-base-uncased` を使用して事前学習済みモデルをベンチマークする代わりに、利用可能な任意のモデルクラスの任意の設定をベンチマークすることもできます。この場合、ベンチマーク引数と共に設定の `list` を挿入する必要があります。


<frameworkcontent>
<pt>
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

>>> args = PyTorchBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = PyTorchBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8              128            0.006
bert-base                  8              512            0.006
bert-base                  8              128            0.018     
bert-base                  8              512            0.088     
bert-384-hid              8               8             0.006     
bert-384-hid              8               32            0.006     
bert-384-hid              8              128            0.011     
bert-384-hid              8              512            0.054     
bert-6-lay                 8               8             0.003     
bert-6-lay                 8               32            0.004     
bert-6-lay                 8              128            0.009     
bert-6-lay                 8              512            0.044
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB 
--------------------------------------------------------------------------------
bert-base                  8               8             1277
bert-base                  8               32            1281
bert-base                  8              128            1307     
bert-base                  8              512            1539     
bert-384-hid              8               8             1005     
bert-384-hid              8               32            1027     
bert-384-hid              8              128            1035     
bert-384-hid              8              512            1255     
bert-6-lay                 8               8             1097     
bert-6-lay                 8               32            1101     
bert-6-lay                 8              128            1127     
bert-6-lay                 8              512            1359
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:35:25.143267
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</pt>
<tf>
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

>>> args = TensorFlowBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = TensorFlowBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8               8             0.005
bert-base                  8               32            0.008
bert-base                  8              128            0.022
bert-base                  8              512            0.106
bert-384-hid              8               8             0.005
bert-384-hid              8               32            0.007
bert-384-hid              8              128            0.018
bert-384-hid              8              512            0.064
bert-6-lay                 8               8             0.002
bert-6-lay                 8               32            0.003
bert-6-lay                 8              128            0.0011
bert-6-lay                 8              512            0.074
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB 
--------------------------------------------------------------------------------
bert-base                  8               8             1330
bert-base                  8               32            1330
bert-base                  8              128            1330
bert-base                  8              512            1770
bert-384-hid              8               8             1330
bert-384-hid              8               32            1330
bert-384-hid              8              128            1330
bert-384-hid              8              512            1540
bert-6-lay                 8               8             1330
bert-6-lay                 8               32            1330
bert-6-lay                 8              128            1330
bert-6-lay                 8              512            1540
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:38:15.487125
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</tf>
</frameworkcontent>

カスタマイズされたBertModelクラスの構成に対する推論時間と必要なメモリのベンチマーク

この機能は、モデルをトレーニングする際にどの構成を選択すべきかを決定する際に特に役立つことがあります。

## Benchmark best practices

このセクションでは、モデルをベンチマークする際に注意すべきいくつかのベストプラクティスをリストアップしています。

- 現在、単一デバイスのベンチマークしかサポートされていません。GPUでベンチマークを実行する場合、コードを実行するデバイスをユーザーが指定することを推奨します。
  これはシェルで`CUDA_VISIBLE_DEVICES`環境変数を設定することで行えます。例：`export CUDA_VISIBLE_DEVICES=0`を実行してからコードを実行します。
- `no_multi_processing`オプションは、テストおよびデバッグ用にのみ`True`に設定すべきです。正確なメモリ計測を確保するために、各メモリベンチマークを別々のプロセスで実行することをお勧めします。これにより、`no_multi_processing`が`True`に設定されます。
- モデルのベンチマーク結果を共有する際には、常に環境情報を記述するべきです。異なるGPUデバイス、ライブラリバージョンなどでベンチマーク結果が大きく異なる可能性があるため、ベンチマーク結果単体ではコミュニティにとってあまり有用ではありません。

## Sharing your benchmark

以前、すべての利用可能なコアモデル（当時10モデル）に対して、多くの異なる設定で推論時間のベンチマークが行われました：PyTorchを使用し、TorchScriptの有無、TensorFlowを使用し、XLAの有無などです。これらのテストはすべてCPUで行われました（TensorFlow XLAを除く）。

このアプローチの詳細については、[次のブログポスト](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2)に詳しく説明されており、結果は[こちら](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing)で利用できます。

新しいベンチマークツールを使用すると、コミュニティとベンチマーク結果を共有することがこれまで以上に簡単になります。

- [PyTorchベンチマーク結果](https://github.com/huggingface/transformers/tree/main/examples/pytorch/benchmarking/README.md)。
- [TensorFlowベンチマーク結果](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/benchmarking/README.md)。

<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Efficient Training on Multiple CPUs

1つのCPUでのトレーニングが遅すぎる場合、複数のCPUを使用できます。このガイドは、PyTorchベースのDDPを使用した分散CPUトレーニングに焦点を当てています。

## Intel® oneCCL Bindings for PyTorch

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL)（集合通信ライブラリ）は、allreduce、allgather、alltoallなどの収集通信を実装した効率的な分散ディープラーニングトレーニング用のライブラリです。oneCCLの詳細については、[oneCCLドキュメント](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)と[oneCCL仕様](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)を参照してください。

モジュール`oneccl_bindings_for_pytorch`（バージョン1.12以前は`torch_ccl`）は、PyTorch C10D ProcessGroup APIを実装し、外部のProcessGroupとして動的にロードでき、現在はLinuxプラットフォームでのみ動作します。

[torch-ccl](https://github.com/intel/torch-ccl)の詳細情報を確認してください。

### Intel® oneCCL Bindings for PyTorch installation:

Wheelファイルは、以下のPythonバージョン用に利用可能です:

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| 1.13.0            |            | √          | √          | √          | √           |
| 1.12.100          |            | √          | √          | √          | √           |
| 1.12.0            |            | √          | √          | √          | √           |
| 1.11.0            |            | √          | √          | √          | √           |
| 1.10.0            | √          | √          | √          | √          |             |

```
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```

where `{pytorch_version}` should be your PyTorch version, for instance 1.13.0.
Check more approaches for [oneccl_bind_pt installation](https://github.com/intel/torch-ccl).
Versions of oneCCL and PyTorch must match.

<Tip warning={true}>

oneccl_bindings_for_pytorch 1.12.0 prebuilt wheel does not work with PyTorch 1.12.1 (it is for PyTorch 1.12.0)
PyTorch 1.12.1 should work with oneccl_bindings_for_pytorch 1.12.100

</Tip>

`{pytorch_version}` は、あなたのPyTorchのバージョン（例：1.13.0）に置き換える必要があります。重要なのは、oneCCLとPyTorchのバージョンが一致していることです。[oneccl_bind_ptのインストール](https://github.com/intel/torch-ccl)に関するさらなるアプローチを確認できます。

<Tip warning={true}>

`oneccl_bindings_for_pytorch`の1.12.0プリビルトホイールはPyTorch 1.12.1と互換性がありません（これはPyTorch 1.12.0用です）。PyTorch 1.12.1を使用する場合は、`oneccl_bindings_for_pytorch`バージョン1.12.100を使用する必要があります。

</Tip>

## Intel® MPI library


この基準ベースのMPI実装を使用して、Intel®アーキテクチャ上で柔軟で効率的、スケーラブルなクラスタメッセージングを提供します。このコンポーネントは、Intel® oneAPI HPC Toolkitの一部です。

oneccl_bindings_for_pytorchはMPIツールセットと一緒にインストールされます。使用する前に環境をソース化する必要があります。


for Intel® oneCCL >= 1.12.0
```
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

for Intel® oneCCL whose version < 1.12.0
```
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### IPEX installation:

IPEXは、Float32およびBFloat16の両方でCPUトレーニングのパフォーマンス最適化を提供します。詳細は[こちらのシングルCPUセクション](./perf_train_cpu)をご参照ください。

以下の「トレーナーでの使用」は、Intel® MPIライブラリでmpirunを使用する例を示しています。

## Usage in Trainer
トレーナーでのマルチCPU分散トレーニングを有効にするために、ユーザーはコマンド引数に **`--ddp_backend ccl`** を追加する必要があります。

例を見てみましょう。[質問応答の例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

以下のコマンドは、1つのXeonノードで2つのプロセスを使用してトレーニングを有効にします。1つのプロセスが1つのソケットで実行されます。OMP_NUM_THREADS/CCL_WORKER_COUNT変数は、最適なパフォーマンスを調整するために調整できます。


```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```

以下のコマンドは、2つのXeonプロセッサ（node0とnode1、node0をメインプロセスとして使用）で合計4つのプロセスを使用してトレーニングを有効にします。ppn（ノードごとのプロセス数）は2に設定され、1つのソケットごとに1つのプロセスが実行されます。最適なパフォーマンスを得るために、OMP_NUM_THREADS/CCL_WORKER_COUNT変数を調整できます。

node0では、各ノードのIPアドレスを含む構成ファイルを作成し、その構成ファイルのパスを引数として渡す必要があります。

```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```

ノード0で次のコマンドを実行すると、ノード0とノード1で**4DDP**がBF16自動混合精度で有効になります。


```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```
<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Training on CPU

このガイドは、CPU上で大規模なモデルを効率的にトレーニングする方法に焦点を当てています。

## Mixed precision with IPEX

IPEXはAVX-512以上のCPUに最適化されており、AVX2のみのCPUでも機能的に動作します。そのため、AVX-512以上のIntel CPU世代ではパフォーマンスの向上が期待されますが、AVX2のみのCPU（例：AMD CPUまたは古いIntel CPU）ではIPEXの下でより良いパフォーマンスが得られるかもしれませんが、保証されません。IPEXは、Float32とBFloat16の両方でCPUトレーニングのパフォーマンスを最適化します。以下のセクションでは、BFloat16の使用に重点を置いて説明します。

低精度データ型であるBFloat16は、AVX512命令セットを備えた第3世代Xeon® Scalable Processors（別名Cooper Lake）でネイティブサポートされており、さらに高性能なIntel® Advanced Matrix Extensions（Intel® AMX）命令セットを備えた次世代のIntel® Xeon® Scalable Processorsでもサポートされます。CPUバックエンド用の自動混合精度がPyTorch-1.10以降で有効になっています。同時に、Intel® Extension for PyTorchでのCPU用BFloat16の自動混合精度サポートと、オペレーターのBFloat16最適化のサポートが大幅に向上し、一部がPyTorchのメインブランチにアップストリームされています。ユーザーはIPEX Auto Mixed Precisionを使用することで、より優れたパフォーマンスとユーザーエクスペリエンスを得ることができます。

詳細な情報については、[Auto Mixed Precision](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html)を確認してください。

### IPEX installation:

IPEXのリリースはPyTorchに従っており、pipを使用してインストールできます：

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 1.13              |  1.13.0+cpu    |
| 1.12              |  1.12.300+cpu  |
| 1.11              |  1.11.200+cpu  |
| 1.10              |  1.10.100+cpu  |

```
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

[IPEXのインストール方法](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html)について、さらなるアプローチを確認してください。

### Trainerでの使用方法
TrainerでIPEXの自動混合精度を有効にするには、ユーザーはトレーニングコマンド引数に `use_ipex`、`bf16`、および `no_cuda` を追加する必要があります。

[Transformersの質問応答](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)のユースケースを例に説明します。

- CPU上でBF16自動混合精度を使用してIPEXでトレーニングを行う場合：
<pre> python run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex \</b>
<b>--bf16 --no_cuda</b></pre>

### Practice example

Blog: [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids)

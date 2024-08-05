<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Efficient Inference on CPU

このガイドは、CPU上で大規模なモデルの効率的な推論に焦点を当てています。

## `BetterTransformer` for faster inference

最近、テキスト、画像、および音声モデルのCPU上での高速な推論のために`BetterTransformer`を統合しました。詳細については、この統合に関するドキュメンテーションを[こちら](https://huggingface.co/docs/optimum/bettertransformer/overview)で確認してください。

## PyTorch JITモード（TorchScript）
TorchScriptは、PyTorchコードからシリアライズ可能で最適化可能なモデルを作成する方法です。任意のTorchScriptプログラムは、Python依存性のないプロセスで保存およびロードできます。
デフォルトのイーガーモードと比較して、PyTorchのjitモードは通常、オペレーターフュージョンなどの最適化手法によりモデル推論のパフォーマンスが向上します。

TorchScriptの簡単な紹介については、[PyTorch TorchScriptチュートリアル](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)を参照してください。

### JITモードでのIPEXグラフ最適化
Intel® Extension for PyTorchは、Transformersシリーズモデルのjitモードにさらなる最適化を提供します。Intel® Extension for PyTorchをjitモードで使用することを強くお勧めします。Transformersモデルからよく使用されるオペレーターパターンのいくつかは、既にIntel® Extension for PyTorchでjitモードのフュージョンに対応しています。これらのフュージョンパターン（Multi-head-attentionフュージョン、Concat Linear、Linear+Add、Linear+Gelu、Add+LayerNormフュージョンなど）は有効でパフォーマンスが良いです。フュージョンの利点は、ユーザーに透過的に提供されます。分析によれば、最も人気のある質問応答、テキスト分類、トークン分類のNLPタスクの約70％が、これらのフュージョンパターンを使用してFloat32精度とBFloat16混合精度の両方でパフォーマンスの利点を得ることができます。

[IPEXグラフ最適化の詳細情報](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)を確認してください。

#### IPEX installation:

IPEXのリリースはPyTorchに従っています。[IPEXのインストール方法](https://intel.github.io/intel-extension-for-pytorch/)を確認してください。

### Usage of JIT-mode
Trainerで評価または予測のためにJITモードを有効にするには、ユーザーはTrainerコマンド引数に`jit_mode_eval`を追加する必要があります。

<Tip warning={true}>

PyTorch >= 1.14.0の場合、jitモードはjit.traceでdict入力がサポートされているため、予測と評価に任意のモデルに利益をもたらす可能性があります。

PyTorch < 1.14.0の場合、jitモードはforwardパラメーターの順序がjit.traceのタプル入力の順序と一致するモデルに利益をもたらす可能性があります（質問応答モデルなど）。jit.traceがタプル入力の順序と一致しない場合、テキスト分類モデルなど、jit.traceは失敗し、これをフォールバックさせるために例外でキャッチしています。ログはユーザーに通知するために使用されます。

</Tip>

[Transformers質問応答の使用例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)を参考にしてください。

- Inference using jit mode on CPU:
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--jit_mode_eval </b></pre> 

- Inference with IPEX using jit mode on CPU:
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--use_ipex \</b>
<b>--jit_mode_eval</b></pre> 

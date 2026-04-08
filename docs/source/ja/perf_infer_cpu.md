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

## PyTorch JITモード（TorchScript）
TorchScriptは、PyTorchコードからシリアライズ可能で最適化可能なモデルを作成する方法です。任意のTorchScriptプログラムは、Python依存性のないプロセスで保存およびロードできます。
デフォルトのイーガーモードと比較して、PyTorchのjitモードは通常、オペレーターフュージョンなどの最適化手法によりモデル推論のパフォーマンスが向上します。

TorchScriptの簡単な紹介については、[PyTorch TorchScriptチュートリアル](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)を参照してください。

### JITモードでのIPEXグラフ最適化
Intel® Extension for PyTorchは、Transformersシリーズモデルのjitモードにさらなる最適化を提供します。Intel® Extension for PyTorchをjitモードで使用することを強くお勧めします。Transformersモデルからよく使用されるオペレーターパターンのいくつかは、既にIntel® Extension for PyTorchでjitモードのフュージョンに対応しています。これらのフュージョンパターン（Multi-head-attentionフュージョン、Concat Linear、Linear+Add、Linear+Gelu、Add+LayerNormフュージョンなど）は有効でパフォーマンスが良いです。フュージョンの利点は、ユーザーに透過的に提供されます。分析によれば、最も人気のある質問応答、テキスト分類、トークン分類のNLPタスクの約70％が、これらのフュージョンパターンを使用してFloat32精度とBFloat16混合精度の両方でパフォーマンスの利点を得ることができます。

[IPEXグラフ最適化の詳細情報](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)を確認してください。

#### IPEX installation:

IPEXのリリースはPyTorchに従っています。[IPEXのインストール方法](https://intel.github.io/intel-extension-for-pytorch/)を確認してください。

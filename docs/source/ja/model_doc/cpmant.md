<!--Copyright 2022 The HuggingFace Team and The OpenBMB Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPMAnt

## Overview

CPM-Ant は、10B パラメータを備えたオープンソースの中国語の事前トレーニング済み言語モデル (PLM) です。これは、CPM-Live のライブ トレーニング プロセスの最初のマイルストーンでもあります。トレーニングプロセスは費用対効果が高く、環境に優しいものです。 CPM-Ant は、CUGE ベンチマークでのデルタ チューニングでも有望な結果を達成しています。フル モデルに加えて、さまざまなハードウェア構成の要件を満たすさまざまな圧縮バージョンも提供しています。 [詳細を見る](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)

このモデルは [OpenBMB](https://huggingface.co/openbmb) によって提供されました。元のコードは [ここ](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live) にあります。

## Resources

- [CPM-Live](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live) に関するチュートリアル。

## CpmAntConfig

[[autodoc]] CpmAntConfig
    - all

## CpmAntTokenizer

[[autodoc]] CpmAntTokenizer
    - all

## CpmAntModel

[[autodoc]] CpmAntModel
    - all
    
## CpmAntForCausalLM

[[autodoc]] CpmAntForCausalLM
    - all
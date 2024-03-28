<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPM

## Overview

CPM モデルは、Zhengyan Zhang、Xu Han、Hao Zhou、Pei Ke、Yuxian Gu によって [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413) で提案されました。葉徳明、秦裕佳、
Yusheng Su、Haozhe Ji、Jian Guan、Fanchao Qi、Xiaozi Wang、Yanan Zheng、Guoyang Zeng、Huanqi Cao、Shengqi Chen、
Daixuan Li、Zhenbo Sun、Zhiyuan Liu、Minlie Huang、Wentao Han、Jie Tang、Juanzi Li、Xiaoyan Zhu、Maosong Sun。

論文の要約は次のとおりです。

*事前トレーニングされた言語モデル (PLM) は、さまざまな下流の NLP タスクに有益であることが証明されています。最近ではGPT-3、
1,750億個のパラメータと570GBの学習データを備え、数回の撮影（1枚でも）の容量で大きな注目を集めました
ゼロショット）学習。ただし、GPT-3 を適用して中国語の NLP タスクに対処することは依然として困難です。
GPT-3 の言語は主に英語であり、パラメーターは公開されていません。この技術レポートでは、
大規模な中国語トレーニング データに対する生成的事前トレーニングを備えた中国語事前トレーニング済み言語モデル (CPM)。最高に
私たちの知識の限りでは、26 億のパラメータと 100GB の中国語トレーニング データを備えた CPM は、事前トレーニングされた中国語としては最大のものです。
言語モデルは、会話、エッセイの作成、
クローゼテストと言語理解。広範な実験により、CPM が多くの環境で優れたパフォーマンスを達成できることが実証されています。
少数ショット (ゼロショットでも) 学習の設定での NLP タスク。*

このモデルは [canwenxu](https://huggingface.co/canwenxu) によって提供されました。オリジナルの実装が見つかります
ここ: https://github.com/TsinghuaAI/CPM-Generate


<Tip>

CPM のアーキテクチャは、トークン化方法を除いて GPT-2 と同じです。詳細については、[GPT-2 ドキュメント](openai-community/gpt2) を参照してください。
API リファレンス情報。

</Tip>

## CpmTokenizer

[[autodoc]] CpmTokenizer

## CpmTokenizerFast

[[autodoc]] CpmTokenizerFast

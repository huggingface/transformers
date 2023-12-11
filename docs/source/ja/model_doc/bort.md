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

# BORT

<Tip warning={true}>

このモデルはメンテナンス モードのみであり、コードを変更する新しい PR は受け付けられません。

このモデルの実行中に問題が発生した場合は、このモデルをサポートしていた最後のバージョン (v4.30.0) を再インストールしてください。
これを行うには、コマンド `pip install -U Transformers==4.30.0` を実行します。

</Tip>

## Overview

BORT モデルは、[Optimal Subarchitecture Extraction for BERT](https://arxiv.org/abs/2010.10499) で提案されました。
Adrian de Wynter and Daniel J. Perry.これは、BERT のアーキテクチャ パラメータの最適なサブセットです。
著者は「ボルト」と呼んでいます。

論文の要約は次のとおりです。

*Devlin らから BERT アーキテクチャのアーキテクチャ パラメータの最適なサブセットを抽出します。 (2018)
ニューラル アーキテクチャ検索のアルゴリズムにおける最近の画期的な技術を適用します。この最適なサブセットを次のように呼びます。
"Bort" は明らかに小さく、有効 (つまり、埋め込み層を考慮しない) サイズは 5.5% です。
オリジナルの BERT 大規模アーキテクチャ、およびネット サイズの 16%。 Bort は 288 GPU 時間で事前トレーニングすることもできます。
最高パフォーマンスの BERT パラメトリック アーキテクチャ バリアントである RoBERTa-large の事前トレーニングに必要な時間の 1.2%
(Liu et al., 2019)、同じマシンで BERT-large をトレーニングするのに必要な GPU 時間の世界記録の約 33%
ハードウェア。また、CPU 上で 7.9 倍高速であるだけでなく、他の圧縮バージョンよりもパフォーマンスが優れています。
アーキテクチャ、および一部の非圧縮バリアント: 0.3% ～ 31% のパフォーマンス向上が得られます。
BERT-large に関して、複数の公開自然言語理解 (NLU) ベンチマークにおける絶対的な評価。*

このモデルは [stefan-it](https://huggingface.co/stefan-it) によって提供されました。元のコードは[ここ](https://github.com/alexa/bort/)にあります。

## Usage tips

- BORT のモデル アーキテクチャは BERT に基づいています。詳細については、[BERT のドキュメント ページ](bert) を参照してください。
  モデルの API リファレンスと使用例。
- BORT は BERT トークナイザーの代わりに RoBERTa トークナイザーを使用します。トークナイザーの API リファレンスと使用例については、[RoBERTa のドキュメント ページ](roberta) を参照してください。
- BORT には、 [Agora](https://adewynter.github.io/notes/bort_algorithms_and_applications.html#fine-tuning-with-algebraic-topology) と呼ばれる特定の微調整アルゴリズムが必要です。
  残念ながらまだオープンソース化されていません。誰かが実装しようとすると、コミュニティにとって非常に役立ちます。
  BORT の微調整を機能させるためのアルゴリズム。
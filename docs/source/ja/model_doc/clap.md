<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLAP

## Overview

CLAP モデルは、[Large Scale Contrastive Language-Audio pretraining with
feature fusion and keyword-to-caption augmentation](https://arxiv.org/pdf/2211.06687.pdf)、Yusong Wu、Ke Chen、Tianyu Zhang、Yuchen Hui、Taylor Berg-Kirkpatrick、Shlomo Dubnov 著。

CLAP (Contrastive Language-Audio Pretraining) は、さまざまな (音声、テキスト) ペアでトレーニングされたニューラル ネットワークです。タスクに合わせて直接最適化することなく、音声が与えられた場合に最も関連性の高いテキスト スニペットを予測するように指示できます。 CLAP モデルは、SWINTransformer を使用して log-Mel スペクトログラム入力からオーディオ特徴を取得し、RoBERTa モデルを使用してテキスト特徴を取得します。次に、テキストとオーディオの両方の特徴が、同じ次元の潜在空間に投影されます。投影されたオーディオとテキストの特徴の間のドット積が、同様のスコアとして使用されます。

論文の要約は次のとおりです。

*対照学習は、マルチモーダル表現学習の分野で目覚ましい成功を収めています。この論文では、音声データと自然言語記述を組み合わせて音声表現を開発する、対照的な言語音声事前トレーニングのパイプラインを提案します。この目標を達成するために、私たちはまず、さまざまなデータ ソースからの 633,526 個の音声とテキストのペアの大規模なコレクションである LAION-Audio-630K をリリースします。次に、さまざまなオーディオ エンコーダとテキスト エンコーダを考慮して、対照的な言語とオーディオの事前トレーニング モデルを構築します。機能融合メカニズムとキーワードからキャプションへの拡張をモデル設計に組み込んで、モデルが可変長の音声入力を処理できるようにし、パフォーマンスを向上させます。 3 番目に、包括的な実験を実行して、テキストから音声への取得、ゼロショット音声分類、教師付き音声分類の 3 つのタスクにわたってモデルを評価します。結果は、私たちのモデルがテキストから音声への検索タスクにおいて優れたパフォーマンスを達成していることを示しています。オーディオ分類タスクでは、モデルはゼロショット設定で最先端のパフォーマンスを達成し、非ゼロショット設定でもモデルの結果に匹敵するパフォーマンスを得ることができます。 LAION-オーディオ-6*

このモデルは、[Younes Belkada](https://huggingface.co/ybelkada) および [Arthur Zucker](https://huggingface.co/ArthurZ) によって提供されました。
元のコードは [こちら](https://github.com/LAION-AI/Clap) にあります。

## ClapConfig

[[autodoc]] ClapConfig
    - from_text_audio_configs

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel
    - forward
    - get_text_features
    - get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel
    - forward

## ClapTextModelWithProjection

[[autodoc]] ClapTextModelWithProjection
    - forward

## ClapAudioModel

[[autodoc]] ClapAudioModel
    - forward

## ClapAudioModelWithProjection

[[autodoc]] ClapAudioModelWithProjection
    - forward
    
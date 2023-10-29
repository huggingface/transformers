<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Audio Spectrogram Transformer

## Overview

オーディオ スペクトログラム トランスフォーマー モデルは、Yuan Gong、Yu-An Chung、James Glass によって [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) で提案されました。
Audio Spectrogram Transformer は、オーディオを画像 (スペクトログラム) に変換することにより、オーディオに [Vision Transformer](vit) を適用します。モデルは最先端の結果を取得します
オーディオ分類用。

論文の要約は次のとおりです。

*過去 10 年間、畳み込みニューラル ネットワーク (CNN) がエンドツーエンドのオーディオ分類モデルの主要な構成要素として広く採用されてきました。これは、オーディオ スペクトログラムから対応するラベルへの直接マッピングを学習することを目的としています。長距離のグローバル コンテキストをより適切に捉えるために、CNN の上にセルフ アテンション メカニズムを追加し、CNN とアテンションのハイブリッド モデルを形成するのが最近の傾向です。ただし、CNN への依存が必要かどうか、純粋に注意に基づくニューラル ネットワークで音声分類で優れたパフォーマンスを得るのに十分かどうかは不明です。このペーパーでは、最初の畳み込みのない、純粋にアテンションベースのオーディオ分類モデルであるオーディオ スペクトログラム トランスフォーマー (AST) を紹介することで質問に答えます。当社はさまざまな音声分類ベンチマークで AST を評価し、AudioSet で 0.485 mAP、ESC-50 で 95.6% の精度、Speech Commands V2 で 98.1% の精度という新しい最先端の結果を達成しました。*

チップ：

- 独自のデータセットでオーディオ スペクトログラム トランスフォーマー (AST) を微調整するときは、入力正規化 (
入力の平均値が 0 で、標準偏差が 0.5 であることを確認してください)。 [`ASTFeatureExtractor`] がこれを処理します。 AudioSet を使用することに注意してください
デフォルトでは平均値と標準値です。 [`ast/src/get_norm_stats.py`](https://github.com/YuanGongND/ast/blob/master/src/get_norm_stats.py) をチェックして、その方法を確認できます。
著者らは下流のデータセットの統計を計算します。
- AST には低い学習率が必要であることに注意してください (著者らは、
[PSLA 論文](https://arxiv.org/abs/2102.01243)) はすぐに収束するため、タスクに適した学習率と学習率スケジューラを検索してください。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/audio_spectogram_transformer_architecture.png"
alt="drawing" width="600"/>

<small> オーディオペクトログラムトランスフォーマーアーキテクチャ。から抜粋 <a href="https://arxiv.org/abs/2104.01778">原紙</a>.</small>

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。
元のコードは [ここ](https://github.com/YuanGongND/ast) にあります。

## Resources

Audio Spectrogram Transformer の使用を開始するのに役立つ、公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

<PipelineTag pipeline="audio-classification"/>

- 音声分類のための AST を使用した推論を説明するノートブックは、[こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/AST) にあります。
- [`ASTForAudioClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)。
- 「オーディオ分類」(../tasks/audio_classification) も参照してください。

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## ASTConfig

[[autodoc]] ASTConfig

## ASTFeatureExtractor

[[autodoc]] ASTFeatureExtractor
    - __call__

## ASTModel

[[autodoc]] ASTModel
    - forward

## ASTForAudioClassification

[[autodoc]] ASTForAudioClassification
    - forward
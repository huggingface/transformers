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

## 概要

Audio Spectrogram Transformerモデルは、「[AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)」という論文でYuan Gong、Yu-An Chung、James Glassによって提案されました。これは、音声を画像（スペクトログラム）に変換することで、音声に[Vision Transformer](vit)を適用します。このモデルは音声分類において最先端の結果を得ています。

論文の要旨は以下の通りです：

*In the past decade, convolutional neural networks (CNNs) have been widely adopted as the main building block for end-to-end audio classification models, which aim to learn a direct mapping from audio spectrograms to corresponding labels. To better capture long-range global context, a recent trend is to add a self-attention mechanism on top of the CNN, forming a CNN-attention hybrid model. However, it is unclear whether the reliance on a CNN is necessary, and if neural networks purely based on attention are sufficient to obtain good performance in audio classification. In this paper, we answer the question by introducing the Audio Spectrogram Transformer (AST), the first convolution-free, purely attention-based model for audio classification. We evaluate AST on various audio classification benchmarks, where it achieves new state-of-the-art results of 0.485 mAP on AudioSet, 95.6% accuracy on ESC-50, and 98.1% accuracy on Speech Commands V2.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/audio_spectogram_transformer_architecture.png"
alt="drawing" width="600"/>

<small> Audio pectrogram Transformerのアーキテクチャ。<a href="https://arxiv.org/abs/2104.01778">元論文</a>より抜粋。</small>

このモデルは[nielsr](https://huggingface.co/nielsr)より提供されました。
オリジナルのコードは[こちら](https://github.com/YuanGongND/ast)で見ることができます。

## 使用上のヒント

- 独自のデータセットでAudio Spectrogram Transformer（AST）をファインチューニングする場合、入力の正規化（入力の平均を0、標準偏差を0.5にすること）処理することが推奨されます。[`ASTFeatureExtractor`]はこれを処理します。デフォルトではAudioSetの平均と標準偏差を使用していることに注意してください。著者が下流のデータセットの統計をどのように計算しているかは、[`ast/src/get_norm_stats.py`](https://github.com/YuanGongND/ast/blob/master/src/get_norm_stats.py)で確認することができます。
- ASTは低い学習率が必要であり（著者は[PSLA論文](https://arxiv.org/abs/2102.01243)で提案されたCNNモデルに比べて10倍小さい学習率を使用しています）、素早く収束するため、タスクに適した学習率と学習率スケジューラーを探すことをお勧めします。

## 参考資料

Audio Spectrogram Transformerの使用を開始するのに役立つ公式のHugging Faceおよびコミュニティ（🌎で示されている）の参考資料の一覧です。

<PipelineTag pipeline="audio-classification"/>

- ASTを用いた音声分類の推論を説明するノートブックは[こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/AST)で見ることができます。
- [`ASTForAudioClassification`]は、この[例示スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification)と[ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)によってサポートされています。
- こちらも参照：[音声分類タスク](../tasks/audio_classification)。

ここに参考資料を提出したい場合は、気兼ねなくPull Requestを開いてください。私たちはそれをレビューいたします！参考資料は、既存のものを複製するのではなく、何か新しいことを示すことが理想的です。

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

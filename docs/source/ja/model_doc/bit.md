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

# Big Transfer (BiT)

## Overview

BiT モデルは、Alexander Kolesnikov、Lucas Beyer、Xiaohua Zhai、Joan Puigcerver、Jessica Yung、Sylvain Gelly によって [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) で提案されました。ニール・ホールズビー。
BiT は、[ResNet](resnet) のようなアーキテクチャ (具体的には ResNetv2) の事前トレーニングをスケールアップするための簡単なレシピです。この方法により、転移学習が大幅に改善されます。

論文の要約は次のとおりです。

*事前トレーニングされた表現の転送により、サンプル効率が向上し、視覚用のディープ ニューラル ネットワークをトレーニングする際のハイパーパラメーター調整が簡素化されます。大規模な教師ありデータセットでの事前トレーニングと、ターゲット タスクでのモデルの微調整のパラダイムを再検討します。私たちは事前トレーニングをスケールアップし、Big Transfer (BiT) と呼ぶシンプルなレシピを提案します。いくつかの慎重に選択されたコンポーネントを組み合わせ、シンプルなヒューリスティックを使用して転送することにより、20 を超えるデータセットで優れたパフォーマンスを実現します。 BiT は、クラスごとに 1 つのサンプルから合計 100 万のサンプルまで、驚くほど広範囲のデータ領域にわたって良好にパフォーマンスを発揮します。 BiT は、ILSVRC-2012 で 87.5%、CIFAR-10 で 99.4%、19 タスクの Visual Task Adaptation Benchmark (VTAB) で 76.3% のトップ 1 精度を達成しました。小規模なデータセットでは、BiT は ILSVRC-2012 (クラスあたり 10 例) で 76.8%、CIFAR-10 (クラスあたり 10 例) で 97.0% を達成しました。高い転写性能を実現する主要成分を詳細に分析※。

## Usage tips

- BiT モデルは、アーキテクチャの点で ResNetv2 と同等ですが、次の点が異なります: 1) すべてのバッチ正規化層が [グループ正規化](https://arxiv.org/abs/1803.08494) に置き換えられます。
2) [重みの標準化](https://arxiv.org/abs/1903.10520) は畳み込み層に使用されます。著者らは、両方の組み合わせが大きなバッチサイズでのトレーニングに役立ち、重要な効果があることを示しています。
転移学習への影響。

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。
元のコードは [こちら](https://github.com/google-research/big_transfer) にあります。

## Resources

BiT を始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

<PipelineTag pipeline="image-classification"/>

- [`BitForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 参照: [画像分類タスク ガイド](../tasks/image_classification)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## BitConfig

[[autodoc]] BitConfig

## BitImageProcessor

[[autodoc]] BitImageProcessor
    - preprocess

## BitModel

[[autodoc]] BitModel
    - forward

## BitForImageClassification

[[autodoc]] BitForImageClassification
    - forward

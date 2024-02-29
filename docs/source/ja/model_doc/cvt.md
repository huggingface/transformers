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

# Convolutional Vision Transformer (CvT)

## Overview

CvT モデルは、Haping Wu、Bin Xiao、Noel Codella、Mengchen Liu、Xiyang Dai、Lu Yuan、Lei Zhang によって [CvT: Introduction Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) で提案されました。畳み込みビジョン トランスフォーマー (CvT) は、ViT に畳み込みを導入して両方の設計の長所を引き出すことにより、[ビジョン トランスフォーマー (ViT)](vit) のパフォーマンスと効率を向上させます。

論文の要約は次のとおりです。

*この論文では、ビジョン トランスフォーマー (ViT) を改善する、畳み込みビジョン トランスフォーマー (CvT) と呼ばれる新しいアーキテクチャを紹介します。
ViT に畳み込みを導入して両方の設計の長所を引き出すことで、パフォーマンスと効率を向上させます。これは次のようにして実現されます。
2 つの主要な変更: 新しい畳み込みトークンの埋め込みを含むトランスフォーマーの階層と、畳み込みトランスフォーマー
畳み込み射影を利用したブロック。これらの変更により、畳み込みニューラル ネットワーク (CNN) の望ましい特性が導入されます。
トランスフォーマーの利点 (動的な注意力、
グローバルなコンテキストとより良い一般化)。私たちは広範な実験を実施することで CvT を検証し、このアプローチが達成できることを示しています。
ImageNet-1k 上の他のビジョン トランスフォーマーや ResNet よりも、パラメータが少なく、FLOP が低い、最先端のパフォーマンスを実現します。加えて、
より大きなデータセット (例: ImageNet-22k) で事前トレーニングし、下流のタスクに合わせて微調整すると、パフォーマンスの向上が維持されます。事前トレーニング済み
ImageNet-22k、当社の CvT-W24 は、ImageNet-1k val set で 87.7\% というトップ 1 の精度を獲得しています。最後に、私たちの結果は、位置エンコーディングが、
既存のビジョン トランスフォーマーの重要なコンポーネントであるこのコンポーネントは、モデルでは安全に削除できるため、高解像度のビジョン タスクの設計が簡素化されます。*

このモデルは [anugunj](https://huggingface.co/anugunj) によって提供されました。元のコードは [ここ](https://github.com/microsoft/CvT) にあります。

## Usage tips

- CvT モデルは通常の Vision Transformer ですが、畳み込みでトレーニングされています。 ImageNet-1K および CIFAR-100 で微調整すると、[オリジナル モデル (ViT)](vit) よりも優れたパフォーマンスを発揮します。
- カスタム データの微調整だけでなく推論に関するデモ ノートブックも [ここ](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) で確認できます ([`ViTFeatureExtractor を置き換えるだけで済みます) `] による [`AutoImageProcessor`] および [`ViTForImageClassification`] による [`CvtForImageClassification`])。
- 利用可能なチェックポイントは、(1) [ImageNet-22k](http://www.image-net.org/) (1,400 万の画像と 22,000 のクラスのコレクション) でのみ事前トレーニングされている、(2) も問題ありません。 ImageNet-22k で調整、または (3) [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/) (ILSVRC 2012 とも呼ばれるコレクション) でも微調整130万の
  画像と 1,000 クラス)。

## Resources

CvT を始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示される) リソースのリスト。

<PipelineTag pipeline="image-classification"/>

- [`CvtForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 参照: [画像分類タスク ガイド](../tasks/image_classification)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## CvtConfig

[[autodoc]] CvtConfig

<frameworkcontent>
<pt>

## CvtModel

[[autodoc]] CvtModel
    - forward

## CvtForImageClassification

[[autodoc]] CvtForImageClassification
    - forward

</pt>
<tf>

## TFCvtModel

[[autodoc]] TFCvtModel
    - call

## TFCvtForImageClassification

[[autodoc]] TFCvtForImageClassification
    - call

</tf>
</frameworkcontent>


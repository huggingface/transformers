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

# ConvNeXt V2

## Overview

ConvNeXt V2 モデルは、Sanghyun Woo、Shobhik Debnath、Ronghang Hu、Xinlei Chen、Zhuang Liu, In So Kweon, Saining Xie. によって [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) で提案されました。
ConvNeXt V2 は、Vision Transformers の設計からインスピレーションを得た純粋な畳み込みモデル (ConvNet) であり、[ConvNeXT](convnext) の後継です。

論文の要約は次のとおりです。

*アーキテクチャの改善と表現学習フレームワークの改善により、視覚認識の分野は 2020 年代初頭に急速な近代化とパフォーマンスの向上を実現しました。たとえば、ConvNeXt に代表される最新の ConvNet は、さまざまなシナリオで強力なパフォーマンスを実証しています。これらのモデルはもともと ImageNet ラベルを使用した教師あり学習用に設計されましたが、マスク オートエンコーダー (MAE) などの自己教師あり学習手法からも潜在的に恩恵を受けることができます。ただし、これら 2 つのアプローチを単純に組み合わせると、パフォーマンスが標準以下になることがわかりました。この論文では、完全畳み込みマスク オートエンコーダ フレームワークと、チャネル間の機能競合を強化するために ConvNeXt アーキテクチャに追加できる新しい Global Response Normalization (GRN) 層を提案します。この自己教師あり学習手法とアーキテクチャの改善の共同設計により、ConvNeXt V2 と呼ばれる新しいモデル ファミリが誕生しました。これにより、ImageNet 分類、COCO 検出、ADE20K セグメンテーションなどのさまざまな認識ベンチマークにおける純粋な ConvNet のパフォーマンスが大幅に向上します。また、ImageNet でトップ 1 の精度 76.7% を誇る効率的な 370 万パラメータの Atto モデルから、最先端の 88.9% を達成する 650M Huge モデルまで、さまざまなサイズの事前トレーニング済み ConvNeXt V2 モデルも提供しています。公開トレーニング データのみを使用した精度*。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png"
alt="描画" width="600"/>

<small> ConvNeXt V2 アーキテクチャ。 <a href="https://arxiv.org/abs/2301.00808">元の論文</a>から抜粋。</small>

このモデルは [adirik](https://huggingface.co/adirik) によって提供されました。元のコードは [こちら](https://github.com/facebookresearch/ConvNeXt-V2) にあります。

## Resources

ConvNeXt V2 の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示される) リソースのリスト。

<PipelineTag pipeline="image-classification"/>

- [`ConvNextV2ForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## ConvNextV2Config

[[autodoc]] ConvNextV2Config

## ConvNextV2Model

[[autodoc]] ConvNextV2Model
    - forward

## ConvNextV2ForImageClassification

[[autodoc]] ConvNextV2ForImageClassification
    - forward

## TFConvNextV2Model

[[autodoc]] TFConvNextV2Model
    - call


## TFConvNextV2ForImageClassification

[[autodoc]] TFConvNextV2ForImageClassification
    - call

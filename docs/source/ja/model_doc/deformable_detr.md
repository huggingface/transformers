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

# Deformable DETR

## Overview

変形可能 DETR モデルは、Xizhou Zhu、Weijie Su、Lewei Lu、Bin Li、Xiaogang Wang, Jifeng Dai によって [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) で提案されました
変形可能な DETR は、参照周囲の少数の主要なサンプリング ポイントのみに注目する新しい変形可能なアテンション モジュールを利用することにより、収束の遅さの問題と元の [DETR](detr) の制限された特徴の空間解像度を軽減します。

論文の要約は次のとおりです。

*DETR は、優れたパフォーマンスを実証しながら、物体検出における多くの手作業で設計されたコンポーネントの必要性を排除するために最近提案されました。ただし、画像特徴マップの処理における Transformer アテンション モジュールの制限により、収束が遅く、特徴の空間解像度が制限されるという問題があります。これらの問題を軽減するために、私たちは Deformable DETR を提案しました。この DETR のアテンション モジュールは、参照周囲の少数の主要なサンプリング ポイントのみに注目します。変形可能な DETR は、10 分の 1 のトレーニング エポックで、DETR よりも優れたパフォーマンス (特に小さなオブジェクトの場合) を達成できます。 COCO ベンチマークに関する広範な実験により、私たちのアプローチの有効性が実証されました。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deformable_detr_architecture.png"
alt="描画" width="600"/>

<small> 変形可能な DETR アーキテクチャ。 <a href="https://arxiv.org/abs/2010.04159">元の論文</a>から抜粋。</small>

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。元のコードは [ここ](https://github.com/fundamentalvision/Deformable-DETR) にあります。

## Usage tips


 - トレーニング Deformable DETR は、元の [DETR](detr) モデルをトレーニングすることと同等です。デモ ノートブックについては、以下の [resources](#resources) セクションを参照してください。

## Resources

Deformable DETR の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示される) リソースのリスト。

<PipelineTag pipeline="object-detection"/>

- [`DeformableDetrForObjectDetection`] のカスタム データセットでの推論と微調整に関するデモ ノートブックは、[こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR) にあります。
- [物体検出タスクガイド](../tasks/object_detection) も参照してください。

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## DeformableDetrImageProcessor

[[autodoc]] DeformableDetrImageProcessor
    - preprocess
    - post_process_object_detection

## DeformableDetrFeatureExtractor

[[autodoc]] DeformableDetrFeatureExtractor
    - __call__
    - post_process_object_detection

## DeformableDetrConfig

[[autodoc]] DeformableDetrConfig

## DeformableDetrModel

[[autodoc]] DeformableDetrModel
    - forward

## DeformableDetrForObjectDetection

[[autodoc]] DeformableDetrForObjectDetection
    - forward

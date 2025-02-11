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

# Conditional DETR

## Overview

条件付き DETR モデルは、[Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152) で Depu Meng、Xiaokang Chen、Zejia Fan、Gang Zeng、Houqiang Li、Yuhui Yuan、Lei Sun,  Jingdong Wang によって提案されました。王京東。条件付き DETR は、高速 DETR トレーニングのための条件付きクロスアテンション メカニズムを提供します。条件付き DETR は DETR よりも 6.7 倍から 10 倍速く収束します。

論文の要約は次のとおりです。

*最近開発された DETR アプローチは、トランスフォーマー エンコーダーおよびデコーダー アーキテクチャを物体検出に適用し、有望なパフォーマンスを実現します。この論文では、トレーニングの収束が遅いという重要な問題を扱い、高速 DETR トレーニングのための条件付きクロスアテンション メカニズムを紹介します。私たちのアプローチは、DETR におけるクロスアテンションが 4 つの四肢の位置特定とボックスの予測にコンテンツの埋め込みに大きく依存しているため、高品質のコンテンツの埋め込みの必要性が高まり、トレーニングの難易度が高くなるという点に動機づけられています。条件付き DETR と呼ばれる私たちのアプローチは、デコーダーのマルチヘッド クロスアテンションのためにデコーダーの埋め込みから条件付きの空間クエリを学習します。利点は、条件付き空間クエリを通じて、各クロスアテンション ヘッドが、個別の領域 (たとえば、1 つのオブジェクトの端またはオブジェクト ボックス内の領域) を含むバンドに注目できることです。これにより、オブジェクト分類とボックス回帰のための個別の領域をローカライズするための空間範囲が狭まり、コンテンツの埋め込みへの依存が緩和され、トレーニングが容易になります。実験結果は、条件付き DETR がバックボーン R50 および R101 で 6.7 倍速く収束し、より強力なバックボーン DC5-R50 および DC5-R101 で 10 倍速く収束することを示しています。コードは https://github.com/Atten4Vis/ConditionalDETR で入手できます。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/conditional_detr_curve.jpg"
alt="描画" width="600"/>

<small> 条件付き DETR は、元の DETR に比べてはるかに速い収束を示します。 <a href="https://arxiv.org/abs/2108.06152">元の論文</a>から引用。</small>

このモデルは [DepuMeng](https://huggingface.co/DepuMeng) によって寄稿されました。元のコードは [ここ](https://github.com/Atten4Vis/ConditionalDETR) にあります。

## Resources

- [オブジェクト検出タスクガイド](../tasks/object_detection)

## ConditionalDetrConfig

[[autodoc]] ConditionalDetrConfig

## ConditionalDetrImageProcessor

[[autodoc]] ConditionalDetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrFeatureExtractor

[[autodoc]] ConditionalDetrFeatureExtractor
    - __call__
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrModel

[[autodoc]] ConditionalDetrModel
    - forward

## ConditionalDetrForObjectDetection

[[autodoc]] ConditionalDetrForObjectDetection
    - forward

## ConditionalDetrForSegmentation

[[autodoc]] ConditionalDetrForSegmentation
    - forward

    
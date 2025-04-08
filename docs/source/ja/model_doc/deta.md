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

# DETA

## Overview

DETA モデルは、[NMS Strikes Back](https://arxiv.org/abs/2212.06137) で Jeffrey Ouyang-Zhang、Jang Hyun Cho、Xingyi Zhou、Philipp Krähenbühl によって提案されました。
DETA (Detection Transformers with Assignment の略) は、1 対 1 の 2 部ハンガリアン マッチング損失を置き換えることにより、[Deformable DETR](deformable_detr) を改善します。
非最大抑制 (NMS) を備えた従来の検出器で使用される 1 対多のラベル割り当てを使用します。これにより、最大 2.5 mAP の大幅な増加が得られます。

論文の要約は次のとおりです。

*Detection Transformer (DETR) は、トレーニング中に 1 対 1 の 2 部マッチングを使用してクエリを一意のオブジェクトに直接変換し、エンドツーエンドのオブジェクト検出を可能にします。最近、これらのモデルは、紛れもない優雅さで COCO の従来の検出器を上回りました。ただし、モデル アーキテクチャやトレーニング スケジュールなど、さまざまな設計において従来の検出器とは異なるため、1 対 1 マッチングの有効性は完全には理解されていません。この研究では、DETR での 1 対 1 のハンガリー語マッチングと、非最大監視 (NMS) を備えた従来の検出器での 1 対多のラベル割り当てとの間の厳密な比較を行います。驚くべきことに、NMS を使用した 1 対多の割り当ては、同じ設定の下で標準的な 1 対 1 のマッチングよりも一貫して優れており、最大 2.5 mAP という大幅な向上が見られます。従来の IoU ベースのラベル割り当てを使用して Deformable-DETR をトレーニングする当社の検出器は、ResNet50 バックボーンを使用して 12 エポック (1x スケジュール) 以内に 50.2 COCO mAP を達成し、この設定で既存のすべての従来の検出器またはトランスベースの検出器を上回りました。複数のデータセット、スケジュール、アーキテクチャに関して、私たちは一貫して、パフォーマンスの高い検出トランスフォーマーには二部マッチングが不要であることを示しています。さらに、検出トランスの成功は、表現力豊かなトランス アーキテクチャによるものであると考えています。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deta_architecture.jpg"
alt="drawing" width="600"/>

<small> DETA の概要。 <a href="https://arxiv.org/abs/2212.06137">元の論文</a>から抜粋。 </small>

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。
元のコードは [ここ](https://github.com/jozhang97/DETA) にあります。

## Resources

DETA の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

- DETA のデモ ノートブックは [こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA) にあります。
- 参照: [オブジェクト検出タスク ガイド](../tasks/object_detection)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## DetaConfig

[[autodoc]] DetaConfig

## DetaImageProcessor

[[autodoc]] DetaImageProcessor
    - preprocess
    - post_process_object_detection

## DetaModel

[[autodoc]] DetaModel
    - forward

## DetaForObjectDetection

[[autodoc]] DetaForObjectDetection
    - forward

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

# Dilated Neighborhood Attention Transformer

## Overview

DiNAT は [Dilated Neighborhood Attender Transformer](https://arxiv.org/abs/2209.15001) で提案されました。
Ali Hassani and Humphrey Shi.

[NAT](nat) を拡張するために、拡張近隣アテンション パターンを追加してグローバル コンテキストをキャプチャします。
そしてそれと比較して大幅なパフォーマンスの向上が見られます。

論文の要約は次のとおりです。

*トランスフォーマーは急速に、さまざまなモダリティにわたって最も頻繁に適用される深層学習アーキテクチャの 1 つになりつつあります。
ドメインとタスク。ビジョンでは、単純なトランスフォーマーへの継続的な取り組みに加えて、階層型トランスフォーマーが
また、そのパフォーマンスと既存のフレームワークへの簡単な統合のおかげで、大きな注目を集めました。
これらのモデルは通常、スライディング ウィンドウの近隣アテンション (NA) などの局所的な注意メカニズムを採用しています。
または Swin Transformer のシフト ウィンドウ セルフ アテンション。自己注意の二次複雑さを軽減するのに効果的ですが、
局所的な注意は、自己注意の最も望ましい 2 つの特性を弱めます。それは、長距離の相互依存性モデリングです。
そして全体的な受容野。このペーパーでは、自然で柔軟で、
NA への効率的な拡張により、よりグローバルなコンテキストを捕捉し、受容野をゼロから指数関数的に拡張することができます。
追加費用。 NA のローカルな注目と DiNA のまばらなグローバルな注目は相互に補完し合うため、私たちは
両方に基づいて構築された新しい階層型ビジョン トランスフォーマーである Dilated Neighborhood Attendant Transformer (DiNAT) を導入します。
DiNAT のバリアントは、NAT、Swin、ConvNeXt などの強力なベースラインに比べて大幅に改善されています。
私たちの大規模モデルは、COCO オブジェクト検出において Swin モデルよりも高速で、ボックス AP が 1.5% 優れています。
COCO インスタンス セグメンテーションでは 1.3% のマスク AP、ADE20K セマンティック セグメンテーションでは 1.1% の mIoU。
新しいフレームワークと組み合わせた当社の大規模バリアントは、COCO (58.2 PQ) 上の新しい最先端のパノプティック セグメンテーション モデルです。
および ADE20K (48.5 PQ)、および Cityscapes (44.5 AP) および ADE20K (35.4 AP) のインスタンス セグメンテーション モデル (追加データなし)。
また、ADE20K (58.2 mIoU) 上の最先端の特殊なセマンティック セグメンテーション モデルとも一致します。
都市景観 (84.5 mIoU) では 2 位にランクされています (追加データなし)。 *


<img
src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dilated-neighborhood-attention-pattern.jpg"
alt="drawing" width="600"/>

<small> 異なる拡張値を使用した近隣アテンション。
<a href="https://arxiv.org/abs/2209.15001">元の論文</a>から抜粋。</small>

このモデルは [Ali Hassani](https://huggingface.co/alihassanijr) によって提供されました。
元のコードは [ここ](https://github.com/SHI-Labs/Neighborhood-Attendance-Transformer) にあります。

## Usage tips

DiNAT は *バックボーン* として使用できます。 「output_hidden_​​states = True」の場合、
`hidden_​​states` と `reshaped_hidden_​​states` の両方を出力します。 `reshape_hidden_​​states` は、`(batch_size, height, width, num_channels)` ではなく、`(batch, num_channels, height, width)` の形状を持っています。

ノート：
- DiNAT は、[NATTEN](https://github.com/SHI-Labs/NATTEN/) による近隣アテンションと拡張近隣アテンションの実装に依存しています。
[shi-labs.com/natten](https://shi-labs.com/natten) を参照して、Linux 用のビルド済みホイールを使用してインストールするか、`pip install natten` を実行してシステム上に構築できます。
後者はコンパイルに時間がかかる可能性があることに注意してください。 NATTEN はまだ Windows デバイスをサポートしていません。
- 現時点ではパッチ サイズ 4 のみがサポートされています。

## Resources

DiNAT の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

<PipelineTag pipeline="image-classification"/>


- [`DinatForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 参照: [画像分類タスク ガイド](../tasks/image_classification)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## DinatConfig

[[autodoc]] DinatConfig

## DinatModel

[[autodoc]] DinatModel
    - forward

## DinatForImageClassification

[[autodoc]] DinatForImageClassification
    - forward

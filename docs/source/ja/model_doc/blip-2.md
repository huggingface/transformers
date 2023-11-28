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

# BLIP-2

## Overview

BLIP-2 モデルは、[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) で提案されました。
Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.・サバレーゼ、スティーブン・ホイ。 BLIP-2 は、軽量の 12 層 Transformer をトレーニングすることで、フリーズされた事前トレーニング済み画像エンコーダーと大規模言語モデル (LLM) を活用します。
それらの間にエンコーダーを配置し、さまざまな視覚言語タスクで最先端のパフォーマンスを実現します。最も注目すべき点は、BLIP-2 が 800 億パラメータ モデルである [Flamingo](https://arxiv.org/abs/2204.14198) を 8.7% 改善していることです。
ゼロショット VQAv2 ではトレーニング可能なパラメーターが 54 分の 1 に減少します。

論文の要約は次のとおりです。

*大規模モデルのエンドツーエンドのトレーニングにより、視覚と言語の事前トレーニングのコストはますます法外なものになってきています。この論文では、市販の凍結済み事前トレーニング画像エンコーダと凍結された大規模言語モデルから視覚言語の事前トレーニングをブートストラップする、汎用的で効率的な事前トレーニング戦略である BLIP-2 を提案します。 BLIP-2 は、2 段階で事前トレーニングされた軽量の Querying Transformer でモダリティのギャップを橋渡しします。最初のステージでは、フリーズされた画像エンコーダーから学習する視覚言語表現をブートストラップします。第 2 段階では、凍結された言語モデルから視覚から言語への生成学習をブートストラップします。 BLIP-2 は、既存の方法よりもトレーニング可能なパラメーターが大幅に少ないにもかかわらず、さまざまな視覚言語タスクで最先端のパフォーマンスを実現します。たとえば、私たちのモデルは、トレーニング可能なパラメーターが 54 分の 1 少ないゼロショット VQAv2 で、Flamingo80B を 8.7% 上回っています。また、自然言語の命令に従うことができる、ゼロショット画像からテキストへの生成というモデルの新しい機能も実証します*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/blip2_architecture.jpg"
alt="drawing" width="600"/> 

<small> BLIP-2 アーキテクチャ。 <a href="https://arxiv.org/abs/2301.12597">元の論文から抜粋。</a> </small>

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。
元のコードは [ここ](https://github.com/salesforce/LAVIS/tree/5ee63d688ba4cebff63acee04adaef2dee9af207) にあります。

## Usage tips

- BLIP-2 は、画像とオプションのテキスト プロンプトを指定して条件付きテキストを生成するために使用できます。推論時には、 [`generate`] メソッドを使用することをお勧めします。
- [`Blip2Processor`] を使用してモデル用の画像を準備し、予測されたトークン ID をデコードしてテキストに戻すことができます。

## Resources

BLIP-2 の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

- 画像キャプション、ビジュアル質問応答 (VQA)、およびチャットのような会話のための BLIP-2 のデモ ノートブックは、[こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BLIP-2) にあります。

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## Blip2Config

[[autodoc]] Blip2Config
    - from_vision_qformer_text_configs

## Blip2VisionConfig

[[autodoc]] Blip2VisionConfig

## Blip2QFormerConfig

[[autodoc]] Blip2QFormerConfig

## Blip2Processor

[[autodoc]] Blip2Processor

## Blip2VisionModel

[[autodoc]] Blip2VisionModel
    - forward

## Blip2QFormerModel

[[autodoc]] Blip2QFormerModel
    - forward

## Blip2Model

[[autodoc]] Blip2Model
    - forward
    - get_text_features
    - get_image_features
    - get_qformer_features

## Blip2ForConditionalGeneration

[[autodoc]] Blip2ForConditionalGeneration
    - forward
    - generate
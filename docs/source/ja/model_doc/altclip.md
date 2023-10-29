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

# AltCLIP

## Overview

AltCLIP モデルは、[AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2) で Zhongzhi Chen、Guang Liu、Bo-Wen Zhang、Fulong Ye、Qinghong によって提案されました。ヤン、レデル・ウー。 AltCLIP
(CLIP での言語エンコーダーの変更) は、さまざまな画像とテキスト、およびテキストとテキストのペアでトレーニングされたニューラル ネットワークです。 CLIPを切り替えることで
テキスト エンコーダと事前トレーニングされた多言語テキスト エンコーダ XLM-R を組み合わせると、ほぼすべてのタスクで CLIP と非常に近いパフォーマンスが得られ、多言語理解などの元の CLIP の機能が拡張されました。

論文の要約は次のとおりです。

*この研究では、強力な二言語マルチモーダル表現モデルをトレーニングするための概念的にシンプルで効果的な方法を紹介します。
OpenAI によってリリースされた事前トレーニング済みマルチモーダル表現モデル CLIP から始めて、そのテキスト エンコーダーを事前トレーニング済みの
多言語テキスト エンコーダ XLM-R を使用し、次の 2 段階のトレーニング スキーマによって言語と画像表現の両方を調整します。
教師学習と対照学習。私たちは、幅広いタスクの評価を通じてメソッドを検証します。新しい最先端を確立します
ImageNet-CN、Flicker30k-CN、COCO-CN などの多数のタスクでのパフォーマンス。さらに、非常に近いパフォーマンスが得られます。
ほぼすべてのタスクで CLIP を使用できるため、CLIP のテキスト エンコーダーを変更するだけで、多言語理解などの拡張機能を実現できることがわかります。

## Usage

AltCLIP の使用法は CLIP と非常に似ています。 CLIP との違いはテキスト エンコーダーです。カジュアルな注意ではなく、双方向の注意を使用していることに注意してください。
そして、XLM-R の [CLS] トークンを使用してテキストの埋め込みを表します。

AltCLIP は、マルチモーダルなビジョンおよび言語モデルです。画像とテキストの類似性やゼロショット画像に使用できます。
分類。 AltCLIP は、ViT のようなトランスフォーマーを使用してビジュアル機能を取得し、双方向言語モデルを使用してテキストを取得します
特徴。次に、テキストと視覚の両方の特徴が、同じ次元の潜在空間に投影されます。ドット
投影された画像とテキストの特徴間の積が同様のスコアとして使用されます。

画像を Transformer エンコーダに供給するために、各画像は固定サイズの重複しないパッチのシーケンスに分割されます。
これらは線形に埋め込まれます。 [CLS] トークンは、イメージ全体の表現として機能するために追加されます。作家たち
また、絶対位置埋め込みを追加し、結果として得られるベクトルのシーケンスを標準の Transformer エンコーダに供給します。
[`CLIPImageProcessor`] を使用して、モデルの画像のサイズ変更 (または再スケール) および正規化を行うことができます。

[`AltCLIPProcessor`] は、[`CLIPImageProcessor`] と [`XLMRobertaTokenizer`] を単一のインスタンスにラップし、両方の機能を実現します。
テキストをエンコードして画像を準備します。次の例は、次のメソッドを使用して画像とテキストの類似性スコアを取得する方法を示しています。
[`AltCLIPProcessor`] と [`AltCLIPModel`]。

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

チップ：

このモデルは`CLIPModel`上に構築されているので、オリジナルのCLIPと同様に使用してください。

このモデルは [jongjyh](https://huggingface.co/jongjyh) によって寄稿されました。

## AltCLIPConfig

[[autodoc]] AltCLIPConfig
    - from_text_vision_configs

## AltCLIPTextConfig

[[autodoc]] AltCLIPTextConfig

## AltCLIPVisionConfig

[[autodoc]] AltCLIPVisionConfig

## AltCLIPProcessor

[[autodoc]] AltCLIPProcessor

## AltCLIPModel

[[autodoc]] AltCLIPModel
    - forward
    - get_text_features
    - get_image_features

## AltCLIPTextModel

[[autodoc]] AltCLIPTextModel
    - forward

## AltCLIPVisionModel

[[autodoc]] AltCLIPVisionModel
    - forward
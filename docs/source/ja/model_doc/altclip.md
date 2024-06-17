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

## 概要


AltCLIPモデルは、「[AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2)」という論文でZhongzhi Chen、Guang Liu、Bo-Wen Zhang、Fulong Ye、Qinghong Yang、Ledell Wuによって提案されました。AltCLIP（CLIPの言語エンコーダーの代替）は、様々な画像-テキストペアおよびテキスト-テキストペアでトレーニングされたニューラルネットワークです。CLIPのテキストエンコーダーを事前学習済みの多言語テキストエンコーダーXLM-Rに置き換えることで、ほぼ全てのタスクでCLIPに非常に近い性能を得られ、オリジナルのCLIPの能力を多言語理解などに拡張しました。

論文の要旨は以下の通りです：

*この研究では、強力なバイリンガルマルチモーダル表現モデルを訓練するための概念的に単純で効果的な方法を提案します。OpenAIによってリリースされたマルチモーダル表現モデルCLIPから開始し、そのテキストエンコーダを事前学習済みの多言語テキストエンコーダXLM-Rに交換し、教師学習と対照学習からなる2段階のトレーニングスキーマを用いて言語と画像の表現を整合させました。幅広いタスクの評価を通じて、我々の方法を検証します。ImageNet-CN、Flicker30k-CN、COCO-CNを含む多くのタスクで新たな最先端の性能を達成しました。さらに、ほぼすべてのタスクでCLIPに非常に近い性能を得ており、これはCLIPのテキストエンコーダを変更するだけで、多言語理解などの拡張を実現できることを示唆しています。*

このモデルは[jongjyh](https://huggingface.co/jongjyh)により提供されました。

## 使用上のヒントと使用例

AltCLIPの使用方法はCLIPに非常に似ています。CLIPとの違いはテキストエンコーダーにあります。私たちはカジュアルアテンションではなく双方向アテンションを使用し、XLM-Rの[CLS]トークンをテキスト埋め込みを表すものとして取ることに留意してください。

AltCLIPはマルチモーダルな視覚言語モデルです。これは画像とテキストの類似度や、ゼロショット画像分類に使用できます。AltCLIPはViTのようなTransformerを使用して視覚的特徴を、双方向言語モデルを使用してテキスト特徴を取得します。テキストと視覚の両方の特徴は、同一の次元を持つ潜在空間に射影されます。射影された画像とテキスト特徴間のドット積が類似度スコアとして使用されます。

Transformerエンコーダーに画像を与えるには、各画像を固定サイズの重複しないパッチの系列に分割し、それらを線形に埋め込みます。画像全体を表現するための[CLS]トークンが追加されます。著者は絶対位置埋め込みも追加し、結果として得られるベクトルの系列を標準的なTransformerエンコーダーに供給します。[`CLIPImageProcessor`]を使用して、モデルのために画像のサイズ変更（または拡大縮小）と正規化を行うことができます。

[`AltCLIPProcessor`]は、テキストのエンコードと画像の前処理を両方行うために、[`CLIPImageProcessor`]と[`XLMRobertaTokenizer`]を単一のインスタンスにラップします。以下の例は、[`AltCLIPProcessor`]と[`AltCLIPModel`]を使用して画像-テキスト類似スコアを取得する方法を示しています。

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

<Tip>

このモデルは`CLIPModel`をベースにしており、オリジナルの[CLIP](clip)と同じように使用してください。

</Tip>

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

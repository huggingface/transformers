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

# Chinese-CLIP

## Overview

Chinese-CLIP An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) で提案されました。周、張周。
Chinese-CLIP は、中国語の画像とテキストのペアの大規模なデータセットに対する CLIP (Radford et al., 2021) の実装です。クロスモーダル検索を実行できるほか、ゼロショット画像分類、オープンドメインオブジェクト検出などのビジョンタスクのビジョンバックボーンとしても機能します。オリジナルの中国語-CLIPコードは[このリンクで](https://github.com/OFA-Sys/Chinese-CLIP)。

論文の要約は次のとおりです。

*CLIP の大成功 (Radford et al., 2021) により、視覚言語の事前訓練のための対照学習の研究と応用が促進されました。この研究では、ほとんどのデータが公開されているデータセットから取得された中国語の画像とテキストのペアの大規模なデータセットを構築し、新しいデータセットで中国語の CLIP モデルを事前トレーニングします。当社では、7,700 万から 9 億 5,800 万のパラメータにわたる、複数のサイズの 5 つの中国 CLIP モデルを開発しています。さらに、モデルのパフォーマンスを向上させるために、最初に画像エンコーダーをフリーズさせてモデルをトレーニングし、次にすべてのパラメーターを最適化してトレーニングする 2 段階の事前トレーニング方法を提案します。私たちの包括的な実験では、中国の CLIP がゼロショット学習と微調整のセットアップで MUGE、Flickr30K-CN、および COCO-CN 上で最先端のパフォーマンスを達成でき、ゼロで競争力のあるパフォーマンスを達成できることを実証しています。 - ELEVATER ベンチマークでの評価に基づくショット画像の分類 (Li et al., 2022)。コード、事前トレーニング済みモデル、デモがリリースされました。*

Chinese-CLIP モデルは、[OFA-Sys](https://huggingface.co/OFA-Sys) によって提供されました。

## Usage example

以下のコード スニペットは、画像とテキストの特徴と類似性を計算する方法を示しています。

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import ChineseCLIPProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> # Squirtle, Bulbasaur, Charmander, Pikachu in English
>>> texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

>>> # compute image feature
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
>>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

>>> # compute text features
>>> inputs = processor(text=texts, padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
>>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

>>> # compute image-text similarity scores
>>> inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
```

現在、次のスケールの事前トレーニング済み Chinese-CLIP モデルが 🤗 Hub で利用可能です。

- [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
- [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
- [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)

## ChineseCLIPConfig

[[autodoc]] ChineseCLIPConfig
    - from_text_vision_configs

## ChineseCLIPTextConfig

[[autodoc]] ChineseCLIPTextConfig

## ChineseCLIPVisionConfig

[[autodoc]] ChineseCLIPVisionConfig

## ChineseCLIPImageProcessor

[[autodoc]] ChineseCLIPImageProcessor
    - preprocess

## ChineseCLIPFeatureExtractor

[[autodoc]] ChineseCLIPFeatureExtractor

## ChineseCLIPProcessor

[[autodoc]] ChineseCLIPProcessor

## ChineseCLIPModel

[[autodoc]] ChineseCLIPModel
    - forward
    - get_text_features
    - get_image_features

## ChineseCLIPTextModel

[[autodoc]] ChineseCLIPTextModel
    - forward

## ChineseCLIPVisionModel

[[autodoc]] ChineseCLIPVisionModel
    - forward
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

# ALIGN

## Overview

ALIGN モデルは、[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) で Chao Jia、yingfei Yang、Ye Xia、Yi-Ting Chen、Zarana によって提案されました。パレク、ヒエウ・ファム、クオック・V・リー、ユンシュアン・スン、ジェン・リー、トム・デューリグ。 ALIGN は、マルチモーダルなビジョンと言語モデルです。画像とテキストの類似性やゼロショット画像の分類に使用できます。 ALIGN は、ビジョン エンコーダとして [EfficientNet](efficientnet)、テキスト エンコーダとして [BERT](bert) を備えたデュアル エンコーダ アーキテクチャを備えており、対比学習によってビジュアル表現とテキスト表現を調整することを学習します。これまでの研究とは異なり、ALIGN は大規模なノイズの多いデータセットを利用し、コーパスのスケールを利用して単純なレシピで SOTA 表現を実現できることを示しています。

論文の要約は次のとおりです。

*事前トレーニングされた表現は、多くの NLP および知覚タスクにとって重要になりつつあります。 NLP での表現学習は、人間による注釈のない生のテキストでのトレーニングに移行しましたが、視覚言語表現と視覚言語表現は依然として、高価な、または専門知識を必要とする精選されたトレーニング データセットに大きく依存しています。ビジョン アプリケーションの場合、表現は主に ImageNet や OpenImages などの明示的なクラス ラベルを持つデータセットを使用して学習されます。視覚言語の場合、Conceptual Captions、MSCOCO、CLIP などの一般的なデータセットはすべて、重要なデータ収集 (およびクリーニング) プロセスを必要とします。このコストのかかるキュレーション プロセスによりデータセットのサイズが制限されるため、トレーニング済みモデルのスケーリングが妨げられます。この論文では、Conceptual Captions データセットで高価なフィルタリングや後処理ステップを行わずに取得された、10 億を超える画像代替テキストのペアからなるノイズの多いデータセットを利用します。シンプルなデュアル エンコーダ アーキテクチャは、コントラスト損失を使用して、画像とテキストのペアの視覚表現と言語表現を調整することを学習します。私たちは、コーパスのスケールがそのノイズを補うことができ、このような単純な学習スキームでも最先端の表現につながることを示します。私たちの視覚表現は、ImageNet や VTAB などの分類タスクに転送されたときに優れたパフォーマンスを実現します。視覚的表現と言語表現が調整されているため、ゼロショット画像分類が可能になり、より洗練されたクロスアテンション モデルと比較した場合でも、Flickr30K および MSCOCO の画像テキスト検索ベンチマークで新しい最先端の結果が得られます。この表現により、複雑なテキストおよびテキスト + 画像クエリによるクロスモダリティ検索も可能になります。*

## Usage

ALIGN は EfficientNet を使用して視覚的な特徴を取得し、BERT を使用してテキストの特徴を取得します。次に、テキストと視覚の両方の特徴が、同じ次元の潜在空間に投影されます。投影された画像とテキストの特徴の間のドット積が類似性スコアとして使用されます。

[`AlignProcessor`] は、[`EfficientNetImageProcessor`] と [`BertTokenizer`] を単一のインスタンスにラップして、テキストのエンコードと画像の前処理の両方を行います。次の例は、[`AlignProcessor`] と [`AlignModel`] を使用して画像とテキストの類似性スコアを取得する方法を示しています。

```python
import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["an image of a cat", "an image of a dog"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)
print(probs)
```

このモデルは [Alara Dirik](https://huggingface.co/adirik) によって提供されました。
元のコードは公開されていません。この実装は元の論文に基づくカカオ ブレイン実装に基づいています。

## Resources

ALIGN の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

- [ALIGN と COYO-700M データセット] に関するブログ投稿 (https://huggingface.co/blog/vit-align)。
- ゼロショット画像分類 [デモ](https://huggingface.co/spaces/adirik/ALIGN-zero-shot-image-classification)。
- `kakaobrain/align-base`モデルの[モデルカード](https://huggingface.co/kakaobrain/align-base)。

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## AlignConfig

[[autodoc]] AlignConfig
    - from_text_vision_configs

## AlignTextConfig

[[autodoc]] AlignTextConfig

## AlignVisionConfig

[[autodoc]] AlignVisionConfig

## AlignProcessor

[[autodoc]] AlignProcessor

## AlignModel

[[autodoc]] AlignModel
    - forward
    - get_text_features
    - get_image_features

## AlignTextModel

[[autodoc]] AlignTextModel
    - forward

## AlignVisionModel

[[autodoc]] AlignVisionModel
    - forward

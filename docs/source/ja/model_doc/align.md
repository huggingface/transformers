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

## 概要

ALIGNモデルは、「[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)」という論文でChao Jia、Yinfei Yang、Ye Xia、Yi-Ting Chen、Zarana Parekh、Hieu Pham、Quoc V. Le、Yunhsuan Sung、Zhen Li、Tom Duerigによって提案されました。ALIGNはマルチモーダルな視覚言語モデルです。これは画像とテキストの類似度や、ゼロショット画像分類に使用できます。ALIGNは[EfficientNet](efficientnet)を視覚エンコーダーとして、[BERT](bert)をテキストエンコーダーとして搭載したデュアルエンコーダー構造を特徴とし、対照学習によって視覚とテキストの表現を整合させることを学びます。それまでの研究とは異なり、ALIGNは巨大でノイジーなデータセットを活用し、コーパスのスケールを利用して単純な方法ながら最先端の表現を達成できることを示しています。

論文の要旨は以下の通りです：

*事前学習された表現は、多くの自然言語処理（NLP）および知覚タスクにとって重要になっています。NLPにおける表現学習は、人間のアノテーションのない生のテキストでの学習へと移行していますが、視覚および視覚言語の表現は依然として精巧な学習データセットに大きく依存しており、これは高価であったり専門知識を必要としたりします。視覚アプリケーションの場合、ImageNetやOpenImagesのような明示的なクラスラベルを持つデータセットを使用して学習されることがほとんどです。視覚言語の場合、Conceptual Captions、MSCOCO、CLIPなどの人気のあるデータセットはすべて、それぞれ無視できないデータ収集（およびクリーニング）プロセスを含みます。このコストのかかるキュレーションプロセスはデータセットのサイズを制限し、訓練されたモデルのスケーリングを妨げます。本論文では、Conceptual Captionsデータセットの高価なフィルタリングや後処理ステップなしで得られた、10億を超える画像alt-textペアのノイズの多いデータセットを活用します。シンプルなデュアルエンコーダーアーキテクチャは、対照損失を使用して画像とテキストペアの視覚的および言語的表現を整合させることを学習します。我々は、コーパスの規模がそのノイズを補い、このような単純な学習スキームでも最先端の表現につながることを示します。我々の視覚表現は、ImageNetやVTABなどの分類タスクへの転移において強力な性能を発揮します。整合した視覚的および言語的表現は、ゼロショット画像分類を可能にし、また、より洗練されたクロスアテンションモデルと比較しても、Flickr30KおよびMSCOCO画像テキスト検索ベンチマークにおいて新たな最先端の結果を達成します。また、これらの表現は、複雑なテキストおよびテキスト+画像のクエリを用いたクロスモーダル検索を可能にします。*

このモデルは[Alara Dirik](https://huggingface.co/adirik)により提供されました。
オリジナルのコードは公開されておらず、この実装は元論文に基づいたKakao Brainの実装をベースにしています。

## 使用例

ALIGNはEfficientNetを使用して視覚的特徴を、BERTを使用してテキスト特徴を取得します。テキストと視覚の両方の特徴は、同一の次元を持つ潜在空間に射影されます。射影された画像とテキスト特徴間のドット積が類似度スコアとして使用されます。

[`AlignProcessor`]は、テキストのエンコードと画像の前処理を両方行うために、[`EfficientNetImageProcessor`]と[`BertTokenizer`]を単一のインスタンスにラップします。以下の例は、[`AlignProcessor`]と[`AlignModel`]を使用して画像-テキスト類似度スコアを取得する方法を示しています。

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

## 参考資料

ALIGNの使用を開始するのに役立つ公式のHugging Faceとコミュニティ（🌎で示されている）の参考資料の一覧です。

- [ALIGNとCOYO-700Mデータセット](https://huggingface.co/blog/vit-align)に関するブログ投稿。
- ゼロショット画像分類[デモ](https://huggingface.co/spaces/adirik/ALIGN-zero-shot-image-classification)。
- `kakaobrain/align-base` モデルの[モデルカード](https://huggingface.co/kakaobrain/align-base)。

ここに参考資料を提出したい場合は、気兼ねなくPull Requestを開いてください。私たちはそれをレビューいたします！参考資料は、既存のものを複製するのではなく、何か新しいことを示すことが理想的です。

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

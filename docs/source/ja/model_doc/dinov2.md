<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DINOv2

## Overview

DINOv2 モデルは、[DINOv2: Learning Robust Visual features without Supervision](https://arxiv.org/abs/2304.07193) で提案されました。
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski.
DINOv2 は、[Vision Transformers](vit) に適用される自己教師ありメソッドである [DINO](https://arxiv.org/abs/2104.14294) のアップグレードです。この方法により、汎用の視覚機能、つまり、微調整することなく画像の配布とタスク全体で機能する機能が可能になります。

論文の要約は次のとおりです。

*大量のデータに対するモデルの事前トレーニングのための自然言語処理における最近の進歩により、コンピューター ビジョンにおける同様の基礎モデルへの道が開かれました。これらのモデルは、万能の視覚機能、つまり、微調整せずに画像の配布やタスク全体で機能する機能を生成することにより、あらゆるシステムでの画像の使用を大幅に簡素化できます。この研究は、既存の事前トレーニング手法、特に自己教師あり手法が、多様なソースから十分に厳選されたデータに基づいてトレーニングされた場合、そのような特徴を生成できることを示しています。既存のアプローチを再考し、さまざまな手法を組み合わせて、データとモデルのサイズに関して事前トレーニングを拡張します。技術的な貢献のほとんどは、大規模なトレーニングの加速と安定化を目的としています。データに関しては、自己教師付き文献で一般的に行われているような、キュレーションされていないデータではなく、専用で多様でキュレーションされた画像データセットを構築するための自動パイプラインを提案します。モデルに関しては、ViT モデル (Dosovitskiy et al., 2020) を 1B パラメーターでトレーニングし、それを利用可能な最高の汎用機能である OpenCLIP (Ilharco et al., 2021) を上回る一連の小さなモデルに抽出します。ほとんどのベンチマークは画像レベルとピクセル レベルで行われます。*

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。
元のコードは [ここ](https://github.com/facebookresearch/dinov2) にあります。

## Usage tips

モデルは`torch.jit.trace`を使用してトレースできます。これは、JIT コンパイルを活用してモデルを最適化し、実行を高速化します。これでもまだいくつかの不一致要素が生成され、元のモデルとトレースされたモデルの差は 1e-4 程度であることに注意してください。

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs[0]

# We have to force return_dict=False for tracing
model.config.return_dict = False

with torch.no_grad():
    traced_model = torch.jit.trace(model, [inputs.pixel_values])
    traced_outputs = traced_model(inputs.pixel_values)

print((last_hidden_states - traced_outputs[0]).abs().max())
```

## Resources

DPT を始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

- DINOv2 のデモ ノートブックは [こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DINOv2) にあります。 🌎

<PipelineTag pipeline="image-classification"/>

- [`Dinov2ForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 参照: [画像分類タスク ガイド](../tasks/image_classification)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## Dinov2Config

[[autodoc]] Dinov2Config

## Dinov2Model

[[autodoc]] Dinov2Model
    - forward

## Dinov2ForImageClassification

[[autodoc]] Dinov2ForImageClassification
    - forward

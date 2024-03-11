<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DePlot

## Overview 

DePlot は、Fangyu Liu、Julian Martin Aisenschlos、Francesco Piccinno、Syrine Krichene、Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun. の論文 [DePlot: One-shot visual language reasoning by plot-to-table translation](https://arxiv.org/abs/2212.10505) で提案されました。パン・

論文の要約には次のように記載されています。

*チャートやプロットなどの視覚言語は人間の世界に遍在しています。プロットやチャートを理解するには、強力な推論スキルが必要です。従来の最先端 (SOTA) モデルには少なくとも数万のトレーニング サンプルが必要であり、その推論能力は、特に人間が作成した複雑なクエリでは依然として大幅に制限されています。この論文では、視覚言語推論に対する最初のワンショット ソリューションを紹介します。私たちは、視覚言語推論の課題を 2 つのステップに分解します。(1) プロットからテキストへの翻訳と、(2) 翻訳されたテキストに対する推論です。この方法の鍵となるのは、プロットまたはチャートの画像を線形化されたテーブルに変換する、DePlot という名前のモダリティ変換モジュールです。その後、DePlot の出力を直接使用して、事前トレーニング済みの大規模言語モデル (LLM) をプロンプトし、LLM の少数ショット推論機能を利用できます。 DePlot を取得するには、統一されたタスク形式とメトリクスを確立することでプロットからテーブルへのタスクを標準化し、このタスクで DePlot をエンドツーエンドでトレーニングします。 DePlot は、プラグアンドプレイ方式で LLM とともに既製で使用できます。 28,000 を超えるデータ ポイントで微調整された SOTA モデルと比較して、ワンショット プロンプトのみを使用する DePlot+LLM は、チャート QA タスクからの人が作成したクエリに関して、微調整された SOTA より 24.0% の改善を達成しました。*

DePlot は、`Pix2Struct` アーキテクチャを使用してトレーニングされたモデルです。 `Pix2Struct` の詳細については、[Pix2Struct ドキュメント](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct) を参照してください。
DePlot は、`Pix2Struct` アーキテクチャの Visual Question Answering サブセットです。入力された質問を画像上にレンダリングし、答えを予測します。

## Usage example

現在、DePlot で使用できるチェックポイントは 1 つです。

- `google/deplot`: ChartQA データセットで微調整された DePlot

```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

## Fine-tuning

DePlot を微調整するには、pix2struct [微調整ノートブック](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb) を参照してください。 `Pix2Struct` モデルの場合、Adafactor とコサイン学習率スケジューラを使用してモデルを微調整すると、収束が高速化されることがわかりました。
```python
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
```

<Tip>

DePlot は、`Pix2Struct`アーキテクチャを使用してトレーニングされたモデルです。 API リファレンスについては、[`Pix2Struct` ドキュメント](pix2struct) を参照してください。

</Tip>
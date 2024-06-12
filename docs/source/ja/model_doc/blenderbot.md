<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Blenderbot

**免責事項:** 何か奇妙なものを見つけた場合は、 [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) を報告してください。

## Overview

Blender チャットボット モデルは、[Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf) Stephen Roller、Emily Dinan、Naman Goyal、Da Ju、Mary Williamson、yinghan Liu、で提案されました。
ジン・シュー、マイル・オット、カート・シャスター、エリック・M・スミス、Y-ラン・ブーロー、ジェイソン・ウェストン、2020年4月30日。

論文の要旨は次のとおりです。

*オープンドメインのチャットボットの構築は、機械学習研究にとって難しい分野です。これまでの研究では次のことが示されていますが、
ニューラル モデルをパラメーターの数とトレーニング対象のデータのサイズでスケーリングすると、結果が向上します。
高性能のチャットボットには他の要素も重要であることを示します。良い会話には多くのことが必要です
会話の専門家がシームレスに融合するスキル: 魅力的な話のポイントを提供し、話を聞く
一貫した態度を維持しながら、知識、共感、個性を適切に表現する
ペルソナ。適切なトレーニング データと選択が与えられた場合、大規模モデルがこれらのスキルを学習できることを示します。
世代戦略。 90M、2.7B、9.4B パラメーター モデルを使用してこれらのレシピのバリアントを構築し、モデルを作成します。
コードは公開されています。人間による評価では、当社の最良のモデルが既存のアプローチよりも優れていることがマルチターンで示されています
魅力と人間性の測定という観点からの対話。次に、分析によってこの作業の限界について説明します。
弊社機種の故障事例*

チップ：

- Blenderbot は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左。

このモデルは [sshleifer](https://huggingface.co/sshleifer) によって提供されました。著者のコードは [ここ](https://github.com/facebookresearch/ParlAI) にあります。

## Implementation Notes

- Blenderbot は、標準の [seq2seq モデル トランスフォーマー](https://arxiv.org/pdf/1706.03762.pdf) ベースのアーキテクチャを使用します。
- 利用可能なチェックポイントは、[モデル ハブ](https://huggingface.co/models?search=blenderbot) で見つけることができます。
- これは *デフォルト* Blenderbot モデル クラスです。ただし、次のような小さなチェックポイントもいくつかあります。
  `facebook/blenderbot_small_90M` はアーキテクチャが異なるため、一緒に使用する必要があります。
  [BlenderbotSmall](ブレンダーボット小)。

## Usage

モデルの使用例を次に示します。

```python
>>> from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

>>> mname = "facebook/blenderbot-400M-distill"
>>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
>>> tokenizer = BlenderbotTokenizer.from_pretrained(mname)
>>> UTTERANCE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
>>> reply_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(reply_ids))
["<s> That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?</s>"]
```

## Documentation resources

- [因果言語モデリング タスク ガイド](../tasks/language_modeling)
- [翻訳タスクガイド](../tasks/translation)
- [要約タスクガイド](../tasks/summarization)

## BlenderbotConfig

[[autodoc]] BlenderbotConfig

## BlenderbotTokenizer

[[autodoc]] BlenderbotTokenizer
    - build_inputs_with_special_tokens

## BlenderbotTokenizerFast

[[autodoc]] BlenderbotTokenizerFast
    - build_inputs_with_special_tokens

## BlenderbotModel

*forward* および *generate* の引数については、`transformers.BartModel`を参照してください。

[[autodoc]] BlenderbotModel
    - forward

## BlenderbotForConditionalGeneration

*forward* と *generate* の引数については、[`~transformers.BartForConditionalGeneration`] を参照してください。

[[autodoc]] BlenderbotForConditionalGeneration
    - forward

## BlenderbotForCausalLM

[[autodoc]] BlenderbotForCausalLM
    - forward

## TFBlenderbotModel

[[autodoc]] TFBlenderbotModel
    - call

## TFBlenderbotForConditionalGeneration

[[autodoc]] TFBlenderbotForConditionalGeneration
    - call

## FlaxBlenderbotModel

[[autodoc]] FlaxBlenderbotModel
    - __call__
    - encode
    - decode

## FlaxBlenderbotForConditionalGeneration

[[autodoc]] FlaxBlenderbotForConditionalGeneration
    - __call__
    - encode
    - decode

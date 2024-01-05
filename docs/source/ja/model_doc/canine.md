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

# CANINE

## Overview

CANINE モデルは、[CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language
Representation](https://arxiv.org/abs/2103.06874)、Jonathan H. Clark、Dan Garrette、Iulia Turc、John Wieting 著。その
明示的なトークン化ステップ (バイト ペアなど) を使用せずに Transformer をトレーニングする最初の論文の 1 つ
エンコーディング (BPE、WordPiece または SentencePiece)。代わりに、モデルは Unicode 文字レベルで直接トレーニングされます。
キャラクターレベルでのトレーニングでは必然的にシーケンスの長さが長くなりますが、CANINE はこれを効率的な方法で解決します。
ディープ Transformer エンコーダを適用する前に、ダウンサンプリング戦略を実行します。

論文の要約は次のとおりです。

*パイプライン NLP システムは、エンドツーエンドのニューラル モデリングに大部分が取って代わられていますが、一般的に使用されているほぼすべてのモデルは
依然として明示的なトークン化手順が必要です。最近のトークン化アプローチはデータ由来のサブワードに基づいていますが、
レキシコンは手動で作成されたトークナイザーよりも脆弱ではありませんが、これらの技術はすべての言語に等しく適しているわけではありません。
言語や固定語彙の使用により、モデルの適応能力が制限される可能性があります。この論文では、CANINE を紹介します。
明示的なトークン化や語彙を使用せずに、文字シーケンスを直接操作するニューラル エンコーダーと、
文字に直接作用するか、オプションでサブワードをソフト誘導バイアスとして使用する事前トレーニング戦略。
よりきめの細かい入力を効果的かつ効率的に使用するために、CANINE はダウンサンプリングを組み合わせて、入力を削減します。
コンテキストをエンコードするディープトランスフォーマースタックを備えたシーケンスの長さ。 CANINE は、同等の mBERT モデルよりも次の点で優れています。
TyDi QA の 2.8 F1 は、モデル パラメータが 28% 少ないにもかかわらず、困難な多言語ベンチマークです。*

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。元のコードは [ここ](https://github.com/google-research/language/tree/master/language/canine) にあります。

## Usage tips

- CANINE は内部で少なくとも 3 つの Transformer エンコーダーを使用します: 2 つの「浅い」エンコーダー (単一のエンコーダーのみで構成)
  レイヤー) と 1 つの「ディープ」エンコーダー (通常の BERT エンコーダー)。まず、「浅い」エンコーダを使用してコンテキストを設定します。
  ローカル アテンションを使用した文字の埋め込み。次に、ダウンサンプリングの後、「ディープ」エンコーダーが適用されます。ついに、
  アップサンプリング後、「浅い」エンコーダを使用して最終的な文字埋め込みが作成されます。アップと
  ダウンサンプリングについては論文に記載されています。
- CANINE は、デフォルトで 2048 文字の最大シーケンス長を使用します。 [`CanineTokenizer`] を使用できます
  モデル用のテキストを準備します。
- 特別な [CLS] トークンの最終的な非表示状態の上に線形レイヤーを配置することで分類を行うことができます。
  (事前定義された Unicode コード ポイントがあります)。ただし、トークン分類タスクの場合は、ダウンサンプリングされたシーケンス
  トークンは、元の文字シーケンスの長さ (2048) と一致するように再度アップサンプリングする必要があります。の
  詳細については、論文を参照してください。

モデルのチェックポイント:

  - [google/canine-c](https://huggingface.co/google/canine-c): 自己回帰文字損失で事前トレーニング済み、
    12 レイヤー、768 隠し、12 ヘッド、121M パラメーター (サイズ ~500 MB)。
  - [google/canine-s](https://huggingface.co/google/canine-s): サブワード損失で事前トレーニング済み、12 層、
    768 個の非表示、12 ヘッド、121M パラメーター (サイズ ~500 MB)。

## Usage example

CANINE は生の文字で動作するため、**トークナイザーなし**で使用できます。

```python
>>> from transformers import CanineModel
>>> import torch

>>> model = CanineModel.from_pretrained("google/canine-c")  # model pre-trained with autoregressive character loss

>>> text = "hello world"
>>> # use Python's built-in ord() function to turn each character into its unicode code point id
>>> input_ids = torch.tensor([[ord(char) for char in text]])

>>> outputs = model(input_ids)  # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

ただし、バッチ推論とトレーニングの場合は、トークナイザーを使用することをお勧めします（すべてをパディング/切り詰めるため）
シーケンスを同じ長さにします):

```python
>>> from transformers import CanineTokenizer, CanineModel

>>> model = CanineModel.from_pretrained("google/canine-c")
>>> tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

>>> inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
>>> encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

>>> outputs = model(**encoding)  # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

## Resources

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

## CanineConfig

[[autodoc]] CanineConfig

## CanineTokenizer

[[autodoc]] CanineTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences

## CANINE specific outputs

[[autodoc]] models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineModel

[[autodoc]] CanineModel
    - forward

## CanineForSequenceClassification

[[autodoc]] CanineForSequenceClassification
    - forward

## CanineForMultipleChoice

[[autodoc]] CanineForMultipleChoice
    - forward

## CanineForTokenClassification

[[autodoc]] CanineForTokenClassification
    - forward

## CanineForQuestionAnswering

[[autodoc]] CanineForQuestionAnswering
    - forward

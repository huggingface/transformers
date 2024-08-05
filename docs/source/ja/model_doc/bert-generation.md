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

# BertGeneration

## Overview

BertGeneration モデルは、次を使用してシーケンス間のタスクに利用できる BERT モデルです。
[Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) で提案されている [`EncoderDecoderModel`]
タスク、Sascha Rothe、Sishi Nagayan、Aliaksei Severyn 著。

論文の要約は次のとおりです。

*大規模なニューラル モデルの教師なし事前トレーニングは、最近、自然言語処理に革命をもたらしました。による
NLP 実践者は、公開されたチェックポイントからウォームスタートして、複数の項目で最先端の技術を推進してきました。
コンピューティング時間を大幅に節約しながらベンチマークを実行します。これまでのところ、主に自然言語に焦点を当ててきました。
タスクを理解する。この論文では、シーケンス生成のための事前トレーニングされたチェックポイントの有効性を実証します。私たちは
公開されている事前トレーニング済み BERT と互換性のある Transformer ベースのシーケンス間モデルを開発しました。
GPT-2 および RoBERTa チェックポイントを使用し、モデルの初期化の有用性について広範な実証研究を実施しました。
エンコーダとデコーダ、これらのチェックポイント。私たちのモデルは、機械翻訳に関する新しい最先端の結果をもたらします。
テキストの要約、文の分割、および文の融合。*

## Usage examples and tips

- モデルを [`EncoderDecoderModel`] と組み合わせて使用​​して、2 つの事前トレーニングされたモデルを活用できます。
  後続の微調整のための BERT チェックポイント。

```python
>>> # leverage checkpoints for Bert2Bert model...
>>> # use BERT's cls token as BOS token and sep token as EOS token
>>> encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
>>> # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
>>> decoder = BertGenerationDecoder.from_pretrained(
...     "google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
... )
>>> bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

>>> # create tokenizer...
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

>>> input_ids = tokenizer(
...     "This is a long article to summarize", add_special_tokens=False, return_tensors="pt"
... ).input_ids
>>> labels = tokenizer("This is a short summary", return_tensors="pt").input_ids

>>> # train...
>>> loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
>>> loss.backward()
```

- 事前トレーニングされた [`EncoderDecoderModel`] もモデル ハブで直接利用できます。

```python
>>> # instantiate sentence fusion model
>>> sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
>>> tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

>>> input_ids = tokenizer(
...     "This is the first sentence. This is the second sentence.", add_special_tokens=False, return_tensors="pt"
... ).input_ids

>>> outputs = sentence_fuser.generate(input_ids)

>>> print(tokenizer.decode(outputs[0]))
```

チップ：

- [`BertGenerationEncoder`] と [`BertGenerationDecoder`] は、
  [`EncoderDecoder`] と組み合わせます。
- 要約、文の分割、文の融合、および翻訳の場合、入力に特別なトークンは必要ありません。
  したがって、入力の末尾に EOS トークンを追加しないでください。

このモデルは、[patrickvonplaten](https://huggingface.co/patrickvonplaten) によって提供されました。元のコードは次のとおりです
[ここ](https://tfhub.dev/s?module-type=text-generation&subtype=module,placeholder) があります。

## BertGenerationConfig

[[autodoc]] BertGenerationConfig

## BertGenerationTokenizer

[[autodoc]] BertGenerationTokenizer
    - save_vocabulary

## BertGenerationEncoder

[[autodoc]] BertGenerationEncoder
    - forward

## BertGenerationDecoder

[[autodoc]] BertGenerationDecoder
    - forward

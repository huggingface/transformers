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

# BARTpho

## Overview

BARTpho モデルは、Nguyen Luong Tran、Duong Minh Le、Dat Quoc Nguyen によって [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnam](https://arxiv.org/abs/2109.09701) で提案されました。

論文の要約は次のとおりです。

*BARTpho には、BARTpho_word と BARTpho_syllable の 2 つのバージョンがあり、初の公開された大規模な単一言語です。
ベトナム語用に事前トレーニングされたシーケンスツーシーケンス モデル。当社の BARTpho は「大規模な」アーキテクチャと事前トレーニングを使用します
シーケンス間ノイズ除去モデル BART のスキームなので、生成 NLP タスクに特に適しています。実験
ベトナム語テキスト要約の下流タスクでは、自動評価と人間による評価の両方で、BARTpho が
強力なベースライン mBART を上回り、最先端の性能を向上させます。将来を容易にするためにBARTphoをリリースします
生成的なベトナム語 NLP タスクの研究と応用。*

このモデルは [dqnguyen](https://huggingface.co/dqnguyen) によって提供されました。元のコードは [こちら](https://github.com/VinAIResearch/BARTpho) にあります。

## Usage example

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

>>> line = "Chúng tôi là những nghiên cứu viên."

>>> input_ids = tokenizer(line, return_tensors="pt")

>>> with torch.no_grad():
...     features = bartpho(**input_ids)  # Models outputs are now tuples

>>> # With TensorFlow 2.0+:
>>> from transformers import TFAutoModel

>>> bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")
>>> input_ids = tokenizer(line, return_tensors="tf")
>>> features = bartpho(**input_ids)
```

## Usage tips

- mBARTに続いて、BARTphoはBARTの「大規模な」アーキテクチャを使用し、その上に追加の層正規化層を備えています。
  エンコーダとデコーダの両方。したがって、[BART のドキュメント](bart) の使用例は、使用に適応する場合に使用されます。
  BARTpho を使用する場合は、BART に特化したクラスを mBART に特化した対応するクラスに置き換えることによって調整する必要があります。
  例えば：

```python
>>> from transformers import MBartForConditionalGeneration

>>> bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
>>> TXT = "Chúng tôi là <mask> nghiên cứu viên."
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = bartpho(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)
>>> print(tokenizer.decode(predictions).split())
```

- この実装はトークン化のみを目的としています。`monolingual_vocab_file`はベトナム語に特化した型で構成されています
  多言語 XLM-RoBERTa から利用できる事前トレーニング済み SentencePiece モデル`vocab_file`から抽出されます。
  他の言語 (サブワードにこの事前トレーニング済み多言語 SentencePiece モデル`vocab_file`を使用する場合)
  セグメンテーションにより、独自の言語に特化した`monolingual_vocab_file`を使用して BartphoTokenizer を再利用できます。

## BartphoTokenizer

[[autodoc]] BartphoTokenizer

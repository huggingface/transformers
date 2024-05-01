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

# BertJapanese

## Overview

BERT モデルは日本語テキストでトレーニングされました。

2 つの異なるトークン化方法を備えたモデルがあります。

- MeCab と WordPiece を使用してトークン化します。これには、[MeCab](https://taku910.github.io/mecab/) のラッパーである [fugashi](https://github.com/polm/fugashi) という追加の依存関係が必要です。
- 文字にトークン化します。

*MecabTokenizer* を使用するには、`pip installTransformers["ja"]` (または、インストールする場合は `pip install -e .["ja"]`) する必要があります。
ソースから）依存関係をインストールします。

[cl-tohakuリポジトリの詳細](https://github.com/cl-tohaku/bert-japanese)を参照してください。

MeCab および WordPiece トークン化でモデルを使用する例:


```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

文字トークン化を使用したモデルの使用例:

```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

<Tip>

- この実装はトークン化方法を除いて BERT と同じです。その他の使用例については、[BERT のドキュメント](bert) を参照してください。

</Tip>

このモデルは[cl-tohaku](https://huggingface.co/cl-tohaku)から提供されました。

## BertJapaneseTokenizer

[[autodoc]] BertJapaneseTokenizer

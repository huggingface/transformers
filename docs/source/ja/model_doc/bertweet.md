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

# BERTweet

## Overview

BERTweet モデルは、Dat Quoc Nguyen、Thanh Vu によって [BERTweet: A pre-trained language model for English Tweets](https://www.aclweb.org/anthology/2020.emnlp-demos.2.pdf) で提案されました。アン・トゥアン・グエンさん。

論文の要約は次のとおりです。

*私たちは、英語ツイート用に初めて公開された大規模な事前トレーニング済み言語モデルである BERTweet を紹介します。私たちのBERTweetは、
BERT ベースと同じアーキテクチャ (Devlin et al., 2019) は、RoBERTa 事前トレーニング手順 (Liu et al.) を使用してトレーニングされます。
al.、2019）。実験では、BERTweet が強力なベースラインである RoBERTa ベースおよび XLM-R ベースを上回るパフォーマンスを示すことが示されています (Conneau et al.,
2020)、3 つのツイート NLP タスクにおいて、以前の最先端モデルよりも優れたパフォーマンス結果が得られました。
品詞タグ付け、固有表現認識およびテキスト分類。*

## Usage example

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

>>> # For transformers v4.x+:
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

>>> # For transformers v3.x:
>>> # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

>>> # INPUT TWEET IS ALREADY NORMALIZED!
>>> line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = bertweet(input_ids)  # Models outputs are now tuples

>>> # With TensorFlow 2.0+:
>>> # from transformers import TFAutoModel
>>> # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
```
<Tip>

この実装は、トークン化方法を除いて BERT と同じです。詳細については、[BERT ドキュメント](bert) を参照してください。
API リファレンス情報。

</Tip>

このモデルは [dqnguyen](https://huggingface.co/dqnguyen) によって提供されました。元のコードは [ここ](https://github.com/VinAIResearch/BERTweet) にあります。

## BertweetTokenizer

[[autodoc]] BertweetTokenizer

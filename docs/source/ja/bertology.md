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


# BERTology

大規模なトランスフォーマー、例えばBERTの内部動作を調査する研究領域が急成長しています（これを「BERTology」とも呼びます）。この分野の良い例は以下です：

- BERT Rediscovers the Classical NLP Pipeline by Ian Tenney, Dipanjan Das, Ellie Pavlick:
  [論文リンク](https://arxiv.org/abs/1905.05950)
- Are Sixteen Heads Really Better than One? by Paul Michel, Omer Levy, Graham Neubig: [論文リンク](https://arxiv.org/abs/1905.10650)
- What Does BERT Look At? An Analysis of BERT's Attention by Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning: [論文リンク](https://arxiv.org/abs/1906.04341)
- CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure: [論文リンク](https://arxiv.org/abs/2210.04633)

この新しい分野の発展を支援するために、BERT/GPT/GPT-2モデルにいくつかの追加機能を組み込み、人々が内部表現にアクセスできるようにしました。これらの機能は、主にPaul Michel氏の優れた研究（[論文リンク](https://arxiv.org/abs/1905.10650)）に基づいています。具体的には、以下の機能が含まれています：

- BERT/GPT/GPT-2のすべての隠れ状態にアクセスすることができます。
- BERT/GPT/GPT-2の各ヘッドの注意重みにアクセスできます。
- ヘッドの出力値と勾配を取得し、ヘッドの重要性スコアを計算し、[論文リンク](https://arxiv.org/abs/1905.10650)で説明されているようにヘッドを削減できます。

これらの機能を理解し、使用するのを支援するために、特定のサンプルスクリプト「[bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py)」を追加しました。このスクリプトは、GLUEで事前トレーニングされたモデルから情報を抽出し、ヘッドを削減する役割を果たします。

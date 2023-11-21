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

# BARThez

## Overview

BARThez モデルは、Moussa Kamal Eddine、Antoine J.-P によって [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) で提案されました。ティクシエ、ミカリス・ヴァジルジャンニス、10月23日、
2020年。

論文の要約:


*帰納的転移学習は、自己教師あり学習によって可能になり、自然言語処理全体を実行します。
(NLP) 分野は、BERT や BART などのモデルにより、無数の自然言語に新たな最先端技術を確立し、嵐を巻き起こしています。
タスクを理解すること。いくつかの注目すべき例外はありますが、利用可能なモデルと研究のほとんどは、
英語を対象に実施されました。この作品では、フランス語用の最初の BART モデルである BARTez を紹介します。
（我々の知る限りに）。 BARThez は、過去の研究から得た非常に大規模な単一言語フランス語コーパスで事前トレーニングされました
BART の摂動スキームに合わせて調整しました。既存の BERT ベースのフランス語モデルとは異なり、
CamemBERT と FlauBERT、BARThez は、エンコーダだけでなく、
そのデコーダは事前トレーニングされています。 FLUE ベンチマークからの識別タスクに加えて、BARThez を新しい評価に基づいて評価します。
この論文とともにリリースする要約データセット、OrangeSum。また、すでに行われている事前トレーニングも継続します。
BARTHez のコーパス上で多言語 BART を事前訓練し、結果として得られるモデル (mBARTHez と呼ぶ) が次のことを示します。
バニラの BARThez を大幅に強化し、CamemBERT や FlauBERT と同等かそれを上回ります。*

このモデルは [moussakam](https://huggingface.co/moussakam) によって寄稿されました。著者のコードは[ここ](https://github.com/moussaKam/BARThez)にあります。

<Tip>

BARThez の実装は、トークン化を除いて BART と同じです。詳細については、[BART ドキュメント](bart) を参照してください。
構成クラスとそのパラメータ。 BARThez 固有のトークナイザーについては以下に記載されています。

</Tip>

### Resources

- BARThez は、BART と同様の方法でシーケンス間のタスクを微調整できます。以下を確認してください。
  [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md)。


## BarthezTokenizer

[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast

[[autodoc]] BarthezTokenizerFast

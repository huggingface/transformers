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

# ConvBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=convbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-convbert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/conv-bert-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

ConvBERT モデルは、[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496) で Zihang Jiang、Weihao Yu、Daquan Zhou、Yunpeng Chen、Jiashi Feng、Shuicheng Yan によって提案されました。
やん。

論文の要約は次のとおりです。

*BERT やそのバリアントなどの事前トレーニング済み言語モデルは、最近、さまざまな環境で目覚ましいパフォーマンスを達成しています。
自然言語理解タスク。ただし、BERT はグローバルな自己注意ブロックに大きく依存しているため、問題が発生します。
メモリ使用量と計算コストが大きくなります。すべての注意が入力シーケンス全体に対してクエリを実行しますが、
グローバルな観点からアテンション マップを生成すると、一部のヘッドはローカルな依存関係のみを学習する必要があることがわかります。
これは、計算の冗長性が存在することを意味します。したがって、我々は、新しいスパンベースの動的畳み込みを提案します。
これらのセルフアテンション ヘッドを置き換えて、ローカルの依存関係を直接モデル化します。新しいコンボリューションヘッドと、
自己注意の頭を休め、グローバルとローカルの両方の状況でより効率的な新しい混合注意ブロックを形成します
学ぶ。この混合注意設計を BERT に装備し、ConvBERT モデルを構築します。実験でわかったことは、
ConvBERT は、トレーニング コストが低く、さまざまな下流タスクにおいて BERT およびその亜種よりも大幅に優れたパフォーマンスを発揮します。
モデルパラメータが少なくなります。注目すべきことに、ConvBERTbase モデルは 86.4 GLUE スコアを達成し、ELECTRAbase よりも 0.7 高いのに対し、
トレーニングコストは 1/4 未満です。コードと事前トレーニングされたモデルがリリースされます。*

このモデルは、[abhishek](https://huggingface.co/abhishek) によって提供されました。オリジナルの実装が見つかります
ここ: https://github.com/yitu-opensource/ConvBert

## Usage tips

ConvBERT トレーニングのヒントは BERT のヒントと似ています。使用上のヒントについては、[BERT ドキュメント](bert) を参照してください。

## Resources

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [マスクされた言語モデリング タスク ガイド](../tasks/masked_lang_modeling)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

## ConvBertConfig

[[autodoc]] ConvBertConfig

## ConvBertTokenizer

[[autodoc]] ConvBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## ConvBertTokenizerFast

[[autodoc]] ConvBertTokenizerFast

<frameworkcontent>
<pt>

## ConvBertModel

[[autodoc]] ConvBertModel
    - forward

## ConvBertForMaskedLM

[[autodoc]] ConvBertForMaskedLM
    - forward

## ConvBertForSequenceClassification

[[autodoc]] ConvBertForSequenceClassification
    - forward

## ConvBertForMultipleChoice

[[autodoc]] ConvBertForMultipleChoice
    - forward

## ConvBertForTokenClassification

[[autodoc]] ConvBertForTokenClassification
    - forward

## ConvBertForQuestionAnswering

[[autodoc]] ConvBertForQuestionAnswering
    - forward

</pt>
<tf>

## TFConvBertModel

[[autodoc]] TFConvBertModel
    - call

## TFConvBertForMaskedLM

[[autodoc]] TFConvBertForMaskedLM
    - call

## TFConvBertForSequenceClassification

[[autodoc]] TFConvBertForSequenceClassification
    - call

## TFConvBertForMultipleChoice

[[autodoc]] TFConvBertForMultipleChoice
    - call

## TFConvBertForTokenClassification

[[autodoc]] TFConvBertForTokenClassification
    - call

## TFConvBertForQuestionAnswering

[[autodoc]] TFConvBertForQuestionAnswering
    - call

</tf>
</frameworkcontent>

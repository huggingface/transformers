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

# DeBERTa-v2

## Overview

DeBERTa モデルは、Pengcheng He、Xiaodong Liu、Jianfeng Gao、Weizhu Chen によって [DeBERTa: Decoding-enhanced BERT with Disentangled Attendant](https://arxiv.org/abs/2006.03654) で提案されました。Google のモデルに基づいています。
2018年にリリースされたBERTモデルと2019年にリリースされたFacebookのRoBERTaモデル。

これは、もつれた注意を解きほぐし、使用されるデータの半分を使用して強化されたマスク デコーダ トレーニングを備えた RoBERTa に基づいて構築されています。
ロベルタ。

論文の要約は次のとおりです。

*事前トレーニングされたニューラル言語モデルの最近の進歩により、多くの自然言語モデルのパフォーマンスが大幅に向上しました。
言語処理 (NLP) タスク。この論文では、新しいモデル アーキテクチャ DeBERTa (Decoding-enhanced BERT with
これは、2 つの新しい技術を使用して BERT モデルと RoBERTa モデルを改善します。 1つ目は、
もつれを解く注意メカニズム。各単語は、その内容をエンコードする 2 つのベクトルを使用して表現され、
単語間の注意の重みは、それらの単語のもつれ解除行列を使用して計算されます。
内容と相対的な位置。 2 番目に、強化されたマスク デコーダを使用して、出力ソフトマックス レイヤを次のように置き換えます。
モデルの事前トレーニング用にマスクされたトークンを予測します。これら 2 つの手法により効率が大幅に向上することを示します。
モデルの事前トレーニングと下流タスクのパフォーマンスの向上。 RoBERTa-Large と比較すると、DeBERTa モデルは半分のレベルでトレーニングされています。
トレーニング データは幅広い NLP タスクで一貫して優れたパフォーマンスを示し、MNLI で +0.9% の改善を達成しました。
(90.2% 対 91.1%)、SQuAD v2.0 では +2.3% (88.4% 対 90.7%)、RACE では +3.6% (83.2% 対 86.8%) でした。 DeBERTa コードと
事前トレーニングされたモデルは https://github.com/microsoft/DeBERTa で公開されます。*

次の情報は、[元の実装で直接表示されます リポジトリ](https://github.com/microsoft/DeBERTa)。 DeBERTa v2 は、DeBERTa モデルの 2 番目のバージョンです。それには以下が含まれます
SuperGLUE 単一モデルの提出に使用された 1.5B モデルは、人間のベースライン 89.8 に対して 89.9 を達成しました。あなたはできる
この投稿に関する詳細については、著者のドキュメントを参照してください。
[ブログ](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/)

v2 の新機能:

- **語彙** v2 では、トレーニング データから構築されたサイズ 128K の新しい語彙を使用するようにトークナイザーが変更されました。
  GPT2 ベースのトークナイザーの代わりに、トークナイザーは
  [sentencepiece ベース](https://github.com/google/sentencepiece) トークナイザー。
- **nGiE(nGram Induced Input Encoding)** DeBERTa-v2 モデルは、最初の畳み込み層とは別に追加の畳み込み層を使用します。
  トランスフォーマー層を使用して、入力トークンのローカル依存関係をよりよく学習します。
- **位置射影行列を注目レイヤーのコンテンツ射影行列と共有** 以前に基づく
  実験では、パフォーマンスに影響を与えることなくパラメータを保存できます。
- **バケットを適用して相対位置をエンコードします** DeBERTa-v2 モデルはログ バケットを使用して相対位置をエンコードします
  T5に似ています。
- **900M モデル & 1.5B モデル** 2 つの追加モデル サイズ: 900M と 1.5B が利用可能で、これにより、パフォーマンスが大幅に向上します。
  下流タスクのパフォーマンス。

このモデルは [DeBERTa](https://huggingface.co/DeBERTa) によって寄稿されました。このモデルの TF 2.0 実装は、
[kamalkraj](https://huggingface.co/kamalkraj) による投稿。元のコードは [こちら](https://github.com/microsoft/DeBERTa) にあります。

## Resources
- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [マスク言語モデリング タスク ガイド](../tasks/masked_language_modeling)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

## DebertaV2Config

[[autodoc]] DebertaV2Config

## DebertaV2Tokenizer

[[autodoc]] DebertaV2Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaV2TokenizerFast

[[autodoc]] DebertaV2TokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

<frameworkcontent>
<pt>

## DebertaV2Model

[[autodoc]] DebertaV2Model
    - forward

## DebertaV2PreTrainedModel

[[autodoc]] DebertaV2PreTrainedModel
    - forward

## DebertaV2ForMaskedLM

[[autodoc]] DebertaV2ForMaskedLM
    - forward

## DebertaV2ForSequenceClassification

[[autodoc]] DebertaV2ForSequenceClassification
    - forward

## DebertaV2ForTokenClassification

[[autodoc]] DebertaV2ForTokenClassification
    - forward

## DebertaV2ForQuestionAnswering

[[autodoc]] DebertaV2ForQuestionAnswering
    - forward

## DebertaV2ForMultipleChoice

[[autodoc]] DebertaV2ForMultipleChoice
    - forward

</pt>
<tf>

## TFDebertaV2Model

[[autodoc]] TFDebertaV2Model
    - call

## TFDebertaV2PreTrainedModel

[[autodoc]] TFDebertaV2PreTrainedModel
    - call

## TFDebertaV2ForMaskedLM

[[autodoc]] TFDebertaV2ForMaskedLM
    - call

## TFDebertaV2ForSequenceClassification

[[autodoc]] TFDebertaV2ForSequenceClassification
    - call

## TFDebertaV2ForTokenClassification

[[autodoc]] TFDebertaV2ForTokenClassification
    - call

## TFDebertaV2ForQuestionAnswering

[[autodoc]] TFDebertaV2ForQuestionAnswering
    - call

## TFDebertaV2ForMultipleChoice

[[autodoc]] TFDebertaV2ForMultipleChoice
    - call

</tf>
</frameworkcontent>



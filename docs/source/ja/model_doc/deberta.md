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

# DeBERTa

## Overview

DeBERTa モデルは、Pengcheng He、Xiaodong Liu、Jianfeng Gao、Weizhu Chen によって [DeBERTa: Decoding-enhanced BERT with Disentangled Attendant](https://huggingface.co/papers/2006.03654) で提案されました。Google のモデルに基づいています。
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


このモデルは [DeBERTa](https://huggingface.co/DeBERTa) によって寄稿されました。このモデルの TF 2.0 実装は、
[kamalkraj](https://huggingface.co/kamalkraj) による寄稿。元のコードは [こちら](https://github.com/microsoft/DeBERTa) にあります。

## Resources

DeBERTa を使い始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示される) リソースのリスト。ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

<PipelineTag pipeline="text-classification"/>

- DeBERTa を使用して [DeepSpeed を使用して大規模モデルのトレーニングを加速する](https://huggingface.co/blog/accelerate-deepspeed) 方法に関するブログ投稿。
- DeBERTa による [機械学習によるスーパーチャージされた顧客サービス](https://huggingface.co/blog/supercharge-customer-service-with-machine-learning) に関するブログ投稿。
- [`DebertaForSequenceClassification`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)。
- [`TFDebertaForSequenceClassification`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)。
- [テキスト分類タスクガイド(英語版)](../../en/tasks/sequence_classification)

<PipelineTag pipeline="token-classification" />

- [`DebertaForTokenClassification`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)。
- [`TFDebertaForTokenClassification`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)。
- [トークン分類](https://huggingface.co/course/chapter7/2?fw=pt) 🤗 ハグフェイスコースの章。
- 🤗 ハグフェイスコースの [バイトペアエンコーディングのトークン化](https://huggingface.co/course/chapter6/5?fw=pt) の章。
- [トークン分類タスクガイド](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`DebertaForMaskedLM`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) でサポートされています。
- [`TFDebertaForMaskedLM`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)。
- [マスクされた言語モデリング](https://huggingface.co/course/chapter7/3?fw=pt) 🤗 顔のハグ コースの章。
- [マスク言語モデリング タスク ガイド](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`DebertaForQuestionAnswering`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)。
- [`TFDebertaForQuestionAnswering`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)。
- [質問回答](https://huggingface.co/course/chapter7/7?fw=pt) 🤗 ハグフェイスコースの章。
- [質問回答タスク ガイド](../tasks/question_answering)

## DebertaConfig

[[autodoc]] DebertaConfig

## DebertaTokenizer

[[autodoc]] DebertaTokenizer
    - get_special_tokens_mask
    - save_vocabulary

## DebertaTokenizerFast

[[autodoc]] DebertaTokenizerFast


## DebertaModel

[[autodoc]] DebertaModel
    - forward

## DebertaPreTrainedModel

[[autodoc]] DebertaPreTrainedModel

## DebertaForMaskedLM

[[autodoc]] DebertaForMaskedLM
    - forward

## DebertaForSequenceClassification

[[autodoc]] DebertaForSequenceClassification
    - forward

## DebertaForTokenClassification

[[autodoc]] DebertaForTokenClassification
    - forward

## DebertaForQuestionAnswering

[[autodoc]] DebertaForQuestionAnswering
    - forward



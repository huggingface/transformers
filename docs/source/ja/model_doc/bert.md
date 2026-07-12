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

# BERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bert-base-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

BERT モデルは、Jacob Devlin、Ming-Wei Chang、Kenton Lee、Kristina Toutanova によって [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://huggingface.co/papers/1810.04805) で提案されました。それは
マスクされた言語モデリング目標と次の文の組み合わせを使用して事前トレーニングされた双方向トランスフォーマー
Toronto Book Corpus と Wikipedia からなる大規模なコーパスでの予測。

論文の要約は次のとおりです。

*BERT と呼ばれる新しい言語表現モデルを導入します。これは Bidirectional Encoder Representations の略です
トランスフォーマーより。最近の言語表現モデルとは異なり、BERT は深い双方向性を事前にトレーニングするように設計されています。
すべてのレイヤーの左と右の両方のコンテキストを共同で条件付けすることにより、ラベルのないテキストから表現します。結果として、
事前トレーニングされた BERT モデルは、出力層を 1 つ追加するだけで微調整して、最先端のモデルを作成できます。
実質的なタスク固有のものを必要とせず、質問応答や言語推論などの幅広いタスクに対応
アーキテクチャの変更。*

*BERT は概念的にはシンプルですが、経験的に強力です。 11 の自然な要素に関する新しい最先端の結果が得られます。
言語処理タスク（GLUE スコアを 80.5% に押し上げる（7.7% ポイントの絶対改善）、MultiNLI を含む）
精度は 86.7% (絶対値 4.6% 向上)、SQuAD v1.1 質問応答テスト F1 は 93.2 (絶対値 1.5 ポイント)
改善) および SQuAD v2.0 テスト F1 から 83.1 (5.1 ポイントの絶対改善)。*

## Usage tips

- BERT は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左。
- BERT は、マスク言語モデリング (MLM) および次の文予測 (NSP) の目標を使用してトレーニングされました。それは
  マスクされたトークンの予測や NLU では一般に効率的ですが、テキスト生成には最適ではありません。
- ランダム マスキングを使用して入力を破壊します。より正確には、事前トレーニング中に、トークンの指定された割合 (通常は 15%) が次によってマスクされます。

    * 確率0.8の特別なマスクトークン
    * 確率 0.1 でマスクされたトークンとは異なるランダムなトークン
    * 確率 0.1 の同じトークン

- モデルは元の文を予測する必要がありますが、2 番目の目的があります。入力は 2 つの文 A と B (間に分離トークンあり) です。確率 50% では、文はコーパス内で連続していますが、残りの 50% では関連性がありません。モデルは、文が連続しているかどうかを予測する必要があります。



このモデルは [thomwolf](https://huggingface.co/thomwolf) によって提供されました。元のコードは [こちら](https://github.com/google-research/bert) にあります。

## Resources

BERT を始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示される) リソースのリスト。ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

<PipelineTag pipeline="text-classification"/>

- に関するブログ投稿 [別の言語での BERT テキスト分類](https://www.philschmid.de/bert-text-classification-in-a-different-language)。
- [マルチラベル テキスト分類のための BERT (およびその友人) の微調整](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb) のノートブック.
-  方法に関するノートブック [PyTorch を使用したマルチラベル分類のための BERT の微調整](https://colab.research.google.com/github/abhmishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)。
- 方法に関するノートブック [要約のために BERT を使用して EncoderDecoder モデルをウォームスタートする](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb)。
- [`BertForSequenceClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)。
- [`TFBertForSequenceClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)。
- [`FlaxBertForSequenceClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)。
- [テキスト分類タスクガイド(英語版)](../../en/tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [Hugging Face Transformers with Keras: Fine-tune a non-English BERT for Named Entity Recognition](https://www.philschmid.de/huggingface-transformers-keras-tf) の使用方法に関するブログ投稿。
- 各単語の最初の単語部分のみを使用した [固有表現認識のための BERT の微調整](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb) のノートブックトークン化中の単語ラベル内。単語のラベルをすべての単語部分に伝播するには、代わりにノートブックのこの [バージョン](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb) を参照してください。
- [`BertForTokenClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)。
- [`TFBertForTokenClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)。
- [`FlaxBertForTokenClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification) によってサポートされています。
- [トークン分類](https://huggingface.co/course/chapter7/2?fw=pt) 🤗 ハグフェイスコースの章。
- [トークン分類タスクガイド](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`BertForMaskedLM`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) でサポートされており、 [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)。
- [`TFBertForMaskedLM`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/lang-modeling#run_mlmpy) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)。
- [`FlaxBertForMaskedLM`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) および [ノートブック]( https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)。
- [マスクされた言語モデリング](https://huggingface.co/course/chapter7/3?fw=pt) 🤗 顔ハグ コースの章。
- [マスクされた言語モデリング タスク ガイド](../tasks/masked_lang_modeling)


<PipelineTag pipeline="question-answering"/>

- [`BertForQuestionAnswering`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)。
- [`TFBertForQuestionAnswering`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)。
- [`FlaxBertForQuestionAnswering`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering) でサポートされています。
- [質問回答](https://huggingface.co/course/chapter7/7?fw=pt) 🤗 ハグフェイスコースの章。
- [質問回答タスク ガイド](../tasks/question_answering)

**複数の選択肢**
- [`BertForMultipleChoice`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)。
- [`TFBertForMultipleChoice`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)。
- [多肢選択タスク ガイド](../tasks/multiple_choice)

⚡️ **推論**
- 方法に関するブログ投稿  [Hugging Face Transformers と AWS Inferentia を使用して BERT 推論を高速化する](https://huggingface.co/blog/bert-inferentia-sagemaker)。
- 方法に関するブログ投稿 [GPU 上の DeepSpeed-Inference を使用して BERT 推論を高速化する](https://www.philschmid.de/bert-deepspeed-inference)。

⚙️ **事前トレーニング**
- [Hugging Face Transformers と Habana Gaudi を使用した BERT の事前トレーニング に関するブログ投稿](https://www.philschmid.de/pre-training-bert-habana)。

🚀 **デプロイ**
- 方法に関するブログ投稿  [ハグフェイス最適化でトランスフォーマーを ONNX に変換する](https://www.philschmid.de/convert-transformers-to-onnx)。
- 方法に関するブログ投稿 [AWS 上の Habana Gaudi を使用したハグ顔トランスフォーマーのための深層学習環境のセットアップ](https://www.philschmid.de/getting-started-habana-gaudi#conclusion)。
- に関するブログ投稿  [Hugging Face Transformers、Amazon SageMaker、および Terraform モジュールを使用した自動スケーリング BERT](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker-advanced)。
- に関するブログ投稿  [HuggingFace、AWS Lambda、Docker を使用したサーバーレス BERT](https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker)。
- に関するブログ投稿 [Amazon SageMaker と Training Compiler を使用した Hugging Face Transformers BERT 微調整](https://www.philschmid.de/huggingface-amazon-sagemaker-training-compiler)。
- に関するブログ投稿  [Transformers と Amazon SageMaker を使用した BERT のタスク固有の知識の蒸留](https://www.philschmid.de/knowledge-distillation-bert-transformers)

## BertConfig

[[autodoc]] BertConfig
    - all

## BertTokenizer

[[autodoc]] BertTokenizer
    - get_special_tokens_mask
    - save_vocabulary

## BertTokenizerLegacy

[[autodoc]] BertTokenizerLegacy

## BertTokenizerFast

[[autodoc]] BertTokenizerFast


## Bert specific outputs

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput


## BertModel

[[autodoc]] BertModel
    - forward

## BertForPreTraining

[[autodoc]] BertForPreTraining
    - forward

## BertLMHeadModel

[[autodoc]] BertLMHeadModel
    - forward

## BertForMaskedLM

[[autodoc]] BertForMaskedLM
    - forward

## BertForNextSentencePrediction

[[autodoc]] BertForNextSentencePrediction
    - forward

## BertForSequenceClassification

[[autodoc]] BertForSequenceClassification
    - forward

## BertForMultipleChoice

[[autodoc]] BertForMultipleChoice
    - forward

## BertForTokenClassification

[[autodoc]] BertForTokenClassification
    - forward

## BertForQuestionAnswering

[[autodoc]] BertForQuestionAnswering
    - forward


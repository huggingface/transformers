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

# ALBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=albert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-albert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/albert-base-v2">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

ALBERT モデルは、[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) で Zhenzhong Lan、Mingda Chen、Sebastian Goodman、Kevin Gimpel、Piyush Sharma によって提案されました。
ラドゥ・ソリカット。メモリ消費量を削減し、トレーニングを増やすための 2 つのパラメータ削減手法を紹介します。
BERT の速度:

- 埋め込み行列を 2 つの小さな行列に分割します。
- グループ間で分割された繰り返しレイヤーを使用します。

論文の要約は次のとおりです。

*自然言語表現を事前トレーニングするときにモデル サイズを増やすと、多くの場合、パフォーマンスが向上します。
下流のタスク。ただし、ある時点で、GPU/TPU メモリの制限により、さらなるモデルの増加が困難になります。
トレーニング時間が長くなり、モデルの予期せぬ劣化が発生します。これらの問題に対処するために、2 つのパラメータ削減を提案します。
メモリ消費量を削減し、BERT のトレーニング速度を向上させる技術。包括的な経験的証拠は次のことを示しています
私たちが提案した方法は、元の BERT と比較してはるかに拡張性の高いモデルにつながることがわかりました。また、
文間の一貫性のモデリングに焦点を当てた自己教師あり損失。それが下流のタスクに一貫して役立つことを示します。
複数の文を入力できます。その結果、当社の最良のモデルは、GLUE、RACE、および
SQuAD ベンチマークでは、BERT-large と比較してパラメーターが少なくなります。*

チップ：

- ALBERT は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左よりも。
- ALBERT は繰り返しレイヤーを使用するため、メモリ使用量は小さくなりますが、計算コストは​​変わりません。
  同じものを反復する必要があるのと同じ数の隠れ層を備えた BERT のようなアーキテクチャに似ています。
  (繰り返し) レイヤーの数。
- 埋め込みサイズ E は、非表示サイズ H justified とは異なります。これは、埋め込みがコンテキストに依存しない (1 つの埋め込みベクトルが 1 つのトークンを表す) のに対し、隠れ状態はコンテキストに依存する (1 つの隠れ状態が一連のトークンを表す) ため、 H > を持つ方が論理的です。 > E. また、埋め込み行列は V x E (V は語彙のサイズ) であるため、大きくなります。 E < H の場合、パラメータは少なくなります。
- レイヤーは、パラメータを共有するグループに分割されます (メモリを節約するため)。
次の文の予測は、文の順序予測によって置き換えられます。入力には、2 つの文 A と B (連続している) があり、A に続いて B、または B に続いて A をフィードします。モデルは、それらが交換されたかどうかを予測する必要があります。か否か。

このモデルは [lysandre](https://huggingface.co/lysandre) によって寄稿されました。このモデルの JAX バージョンは次の寄稿者によって提供されました。
[カマルクラジ](https://huggingface.co/kamalkraj)。元のコードは [こちら](https://github.com/google-research/ALBERT) にあります。

## Documentation resources

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [マスクされた言語モデリング タスク ガイド](../tasks/masked_lang_modeling)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

## AlbertConfig

[[autodoc]] AlbertConfig

## AlbertTokenizer

[[autodoc]] AlbertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## AlbertTokenizerFast

[[autodoc]] AlbertTokenizerFast

## Albert specific outputs

[[autodoc]] models.albert.modeling_albert.AlbertForPreTrainingOutput

[[autodoc]] models.albert.modeling_tf_albert.TFAlbertForPreTrainingOutput

## AlbertModel

[[autodoc]] AlbertModel
    - forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining
    - forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM
    - forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification
    - forward

## AlbertForMultipleChoice

[[autodoc]] AlbertForMultipleChoice

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification
    - forward

## AlbertForQuestionAnswering

[[autodoc]] AlbertForQuestionAnswering
    - forward

## TFAlbertModel

[[autodoc]] TFAlbertModel
    - call

## TFAlbertForPreTraining

[[autodoc]] TFAlbertForPreTraining
    - call

## TFAlbertForMaskedLM

[[autodoc]] TFAlbertForMaskedLM
    - call

## TFAlbertForSequenceClassification

[[autodoc]] TFAlbertForSequenceClassification
    - call

## TFAlbertForMultipleChoice

[[autodoc]] TFAlbertForMultipleChoice
    - call

## TFAlbertForTokenClassification

[[autodoc]] TFAlbertForTokenClassification
    - call

## TFAlbertForQuestionAnswering

[[autodoc]] TFAlbertForQuestionAnswering
    - call

## FlaxAlbertModel

[[autodoc]] FlaxAlbertModel
    - __call__

## FlaxAlbertForPreTraining

[[autodoc]] FlaxAlbertForPreTraining
    - __call__

## FlaxAlbertForMaskedLM

[[autodoc]] FlaxAlbertForMaskedLM
    - __call__

## FlaxAlbertForSequenceClassification

[[autodoc]] FlaxAlbertForSequenceClassification
    - __call__

## FlaxAlbertForMultipleChoice

[[autodoc]] FlaxAlbertForMultipleChoice
    - __call__

## FlaxAlbertForTokenClassification

[[autodoc]] FlaxAlbertForTokenClassification
    - __call__

## FlaxAlbertForQuestionAnswering

[[autodoc]] FlaxAlbertForQuestionAnswering
    - __call__


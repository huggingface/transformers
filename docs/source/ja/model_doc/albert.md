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

## 概要

ALBERTモデルは、「[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)」という論文でZhenzhong Lan、Mingda Chen、Sebastian Goodman、Kevin Gimpel、Piyush Sharma、Radu Soricutによって提案されました。BERTのメモリ消費を減らしトレーニングを高速化するためのパラメータ削減技術を2つ示しています：

- 埋め込み行列を2つの小さな行列に分割する。
- グループ間で分割された繰り返し層を使用する。

論文の要旨は以下の通りです：

*自然言語表現の事前学習時にモデルのサイズを増やすと、下流タスクのパフォーマンスが向上することがしばしばあります。しかし、ある時点でさらなるモデルの増大は、GPU/TPUのメモリ制限、長い訓練時間、予期せぬモデルの劣化といった問題のために困難になります。これらの問題に対処するために、我々はBERTのメモリ消費を低減し、訓練速度を高めるための2つのパラメータ削減技術を提案します。包括的な実証的証拠は、我々の提案方法が元のBERTに比べてはるかによくスケールするモデルを生み出すことを示しています。また、文間の一貫性をモデリングに焦点を当てた自己教師あり損失を使用し、複数の文が含まれる下流タスクに一貫して助けとなることを示します。その結果、我々の最良のモデルは、BERT-largeに比べてパラメータが少ないにもかかわらず、GLUE、RACE、SQuADベンチマークで新たな最先端の結果を確立します。*

このモデルは[lysandre](https://huggingface.co/lysandre)により提供されました。このモデルのjaxバージョンは[kamalkraj](https://huggingface.co/kamalkraj)により提供されました。オリジナルのコードは[こちら](https://github.com/google-research/ALBERT)で見ることができます。

## 使用上のヒント

- ALBERTは絶対位置埋め込みを使用するモデルなので、通常、入力を左側ではなく右側にパディングすることが推奨されます。
- ALBERTは繰り返し層を使用するためメモリ使用量は小さくなりますが、同じ数の（繰り返し）層を反復しなければならないため、隠れ層の数が同じであればBERTのようなアーキテクチャと同様の計算コストがかかります。
- 埋め込みサイズEは隠れサイズHと異なりますが、これは埋め込みが文脈に依存しない（一つの埋め込みベクトルが一つのトークンを表す）のに対し、隠れ状態は文脈に依存する（1つの隠れ状態がトークン系列を表す）ため、H >> Eとすることがより論理的です。また、埋め込み行列のサイズはV x Eと大きいです（Vは語彙サイズ）。E < Hであれば、パラメータは少なくなります。
- 層はパラメータを共有するグループに分割されています（メモリ節約のため）。次文予測（NSP: Next Sentence Prediction）は文の順序予測に置き換えられます：入力では、2つの文AとB（それらは連続している）があり、Aに続いてBを与えるか、Bに続いてAを与えます。モデルはそれらが入れ替わっているかどうかを予測する必要があります。

## 参考資料

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問応答タスクガイド](../tasks/question_answering)
- [マスクされた言語モデルタスクガイド](../tasks/masked_language_modeling)
- [多肢選択タスクガイド](../tasks/multiple_choice)

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

<frameworkcontent>
<pt>

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

</pt>

<tf>

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

</tf>
<jax>

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

</jax>
</frameworkcontent>

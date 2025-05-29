<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BigBird

## Overview

BigBird モデルは、[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) で提案されました。
ザヒール、マンジルとグルガネシュ、グルとダベイ、クマール・アヴィナヴァとエインズリー、ジョシュアとアルベルティ、クリスとオンタノン、
サンティアゴとファム、フィリップとラブラ、アニルードとワン、キーファンとヤン、リーなど。 BigBird は注目度が低い
BERT などの Transformer ベースのモデルをさらに長いシーケンスに拡張する、Transformer ベースのモデル。まばらに加えて
アテンションと同様に、BigBird は入力シーケンスにランダム アテンションだけでなくグローバル アテンションも適用します。理論的には、
まばらで全体的でランダムな注意を適用すると、完全な注意に近づくことが示されていますが、
長いシーケンスでは計算効率が大幅に向上します。より長いコンテキストを処理できる機能の結果として、
BigBird は、質問応答や
BERT または RoBERTa と比較した要約。

論文の要約は次のとおりです。

*BERT などのトランスフォーマーベースのモデルは、NLP で最も成功した深層学習モデルの 1 つです。
残念ながら、それらの中核的な制限の 1 つは、シーケンスに対する二次依存性 (主にメモリに関する) です。
完全な注意メカニズムによる長さです。これを解決するために、BigBird は、まばらな注意メカニズムを提案します。
この二次依存関係を線形に削減します。 BigBird がシーケンス関数の汎用近似器であることを示します。
チューリングは完全であるため、二次完全注意モデルのこれらの特性が保存されます。途中、私たちの
理論分析により、O(1) 個のグローバル トークン (CLS など) を持つ利点の一部が明らかになり、
スパース注意メカニズムの一部としてのシーケンス。提案されたスパース アテンションは、次の長さのシーケンスを処理できます。
同様のハードウェアを使用して以前に可能であったものの 8 倍。より長いコンテキストを処理できる機能の結果として、
BigBird は、質問応答や要約などのさまざまな NLP タスクのパフォーマンスを大幅に向上させます。私達も
ゲノミクスデータへの新しいアプリケーションを提案します。*

チップ：

- BigBird の注意がどのように機能するかについての詳細な説明については、[このブログ投稿](https://huggingface.co/blog/big-bird) を参照してください。
- BigBird には、**original_full** と **block_sparse** の 2 つの実装が付属しています。シーケンス長が 1024 未満の場合、次を使用します。
  **block_sparse** を使用してもメリットがないため、**original_full** を使用することをお勧めします。
- コードは現在、3 ブロックと 2 グローバル ブロックのウィンドウ サイズを使用しています。
- シーケンスの長さはブロック サイズで割り切れる必要があります。
- 現在の実装では **ITC** のみがサポートされています。
- 現在の実装では **num_random_blocks = 0** はサポートされていません
- BigBird は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左。

  このモデルは、[vasudevgupta](https://huggingface.co/vasudevgupta) によって提供されました。元のコードが見つかる
[こちら](https://github.com/google-research/bigbird)。

## ドキュメント リソース

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [因果言語モデリング タスク ガイド](../tasks/language_modeling)
- [マスクされた言語モデリング タスク ガイド](../tasks/masked_lang_modeling)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

## BigBirdConfig

[[autodoc]] BigBirdConfig

## BigBirdTokenizer

[[autodoc]] BigBirdTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BigBirdTokenizerFast

[[autodoc]] BigBirdTokenizerFast

## BigBird specific outputs

[[autodoc]] models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput

<frameworkcontent>
<pt>

## BigBirdModel

[[autodoc]] BigBirdModel
    - forward

## BigBirdForPreTraining

[[autodoc]] BigBirdForPreTraining
    - forward

## BigBirdForCausalLM

[[autodoc]] BigBirdForCausalLM
    - forward

## BigBirdForMaskedLM

[[autodoc]] BigBirdForMaskedLM
    - forward

## BigBirdForSequenceClassification

[[autodoc]] BigBirdForSequenceClassification
    - forward

## BigBirdForMultipleChoice

[[autodoc]] BigBirdForMultipleChoice
    - forward

## BigBirdForTokenClassification

[[autodoc]] BigBirdForTokenClassification
    - forward

## BigBirdForQuestionAnswering

[[autodoc]] BigBirdForQuestionAnswering
    - forward

</pt>
<jax>

## FlaxBigBirdModel

[[autodoc]] FlaxBigBirdModel
    - __call__

## FlaxBigBirdForPreTraining

[[autodoc]] FlaxBigBirdForPreTraining
    - __call__

## FlaxBigBirdForCausalLM

[[autodoc]] FlaxBigBirdForCausalLM
    - __call__

## FlaxBigBirdForMaskedLM

[[autodoc]] FlaxBigBirdForMaskedLM
    - __call__

## FlaxBigBirdForSequenceClassification

[[autodoc]] FlaxBigBirdForSequenceClassification
    - __call__

## FlaxBigBirdForMultipleChoice

[[autodoc]] FlaxBigBirdForMultipleChoice
    - __call__

## FlaxBigBirdForTokenClassification

[[autodoc]] FlaxBigBirdForTokenClassification
    - __call__

## FlaxBigBirdForQuestionAnswering

[[autodoc]] FlaxBigBirdForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>


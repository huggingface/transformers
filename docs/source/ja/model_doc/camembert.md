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

# CamemBERT

## Overview

CamemBERT モデルは、[CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) で提案されました。
Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la
Clergerie, Djamé Seddah, and Benoît Sagot. 2019年にリリースされたFacebookのRoBERTaモデルをベースにしたモデルです。
138GBのフランス語テキストでトレーニングされました。

論文の要約は次のとおりです。

*事前トレーニングされた言語モデルは現在、自然言語処理で広く普及しています。成功にもかかわらず、利用可能なほとんどの
モデルは英語のデータ、または複数言語のデータの連結でトレーニングされています。これにより、
このようなモデルの実際の使用は、英語を除くすべての言語で非常に限られています。フランス人にとってこの問題に対処することを目指して、
Bi-direction Encoders for Transformers (BERT) のフランス語版である CamemBERT をリリースします。測定します
複数の下流タスク、つまり品詞タグ付けにおける多言語モデルと比較した CamemBERT のパフォーマンス
依存関係解析、固有表現認識、自然言語推論。 CamemBERT は最先端技術を向上させます
検討されているほとんどのタスクに対応します。私たちは、研究と
フランス語 NLP の下流アプリケーション。*

このモデルは [camembert](https://huggingface.co/camembert) によって提供されました。元のコードは [ここ](https://camembert-model.fr/) にあります。


<Tip>

この実装はRoBERTaと同じです。使用例については[RoBERTaのドキュメント](roberta)も参照してください。
入力と出力に関する情報として。

</Tip>

## Resources

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [因果言語モデリング タスク ガイド](../tasks/language_modeling)
- [マスク言語モデリング タスク ガイド](../tasks/masked_language_modeling)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

## CamembertConfig

[[autodoc]] CamembertConfig

## CamembertTokenizer

[[autodoc]] CamembertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CamembertTokenizerFast

[[autodoc]] CamembertTokenizerFast

<frameworkcontent>
<pt>

## CamembertModel

[[autodoc]] CamembertModel

## CamembertForCausalLM

[[autodoc]] CamembertForCausalLM

## CamembertForMaskedLM

[[autodoc]] CamembertForMaskedLM

## CamembertForSequenceClassification

[[autodoc]] CamembertForSequenceClassification

## CamembertForMultipleChoice

[[autodoc]] CamembertForMultipleChoice

## CamembertForTokenClassification

[[autodoc]] CamembertForTokenClassification

## CamembertForQuestionAnswering

[[autodoc]] CamembertForQuestionAnswering

</pt>
<tf>

## TFCamembertModel

[[autodoc]] TFCamembertModel

## TFCamembertForCasualLM

[[autodoc]] TFCamembertForCausalLM

## TFCamembertForMaskedLM

[[autodoc]] TFCamembertForMaskedLM

## TFCamembertForSequenceClassification

[[autodoc]] TFCamembertForSequenceClassification

## TFCamembertForMultipleChoice

[[autodoc]] TFCamembertForMultipleChoice

## TFCamembertForTokenClassification

[[autodoc]] TFCamembertForTokenClassification

## TFCamembertForQuestionAnswering

[[autodoc]] TFCamembertForQuestionAnswering

</tf>
</frameworkcontent>

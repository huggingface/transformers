<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Data2Vec

## Overview

Data2Vec モデルは、[data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/pdf/2202.03555) で Alexei Baevski、Wei-Ning Hsu、Qiantong Xu、バArun Babu, Jiatao Gu and Michael Auli.
Data2Vec は、テキスト、音声、画像などのさまざまなデータ モダリティにわたる自己教師あり学習のための統一フレームワークを提案します。
重要なのは、事前トレーニングの予測ターゲットは、モダリティ固有のコンテキストに依存しないターゲットではなく、入力のコンテキスト化された潜在表現であることです。

論文の要約は次のとおりです。

*自己教師あり学習の一般的な考え方はどのモダリティでも同じですが、実際のアルゴリズムと
単一のモダリティを念頭に置いて開発されたため、目的は大きく異なります。一般に近づけるために
自己教師あり学習では、どちらの音声に対しても同じ学習方法を使用するフレームワークである data2vec を紹介します。
NLP またはコンピューター ビジョン。中心となるアイデアは、完全な入力データの潜在的な表現を、
標準の Transformer アーキテクチャを使用した自己蒸留セットアップの入力のマスクされたビュー。
単語、視覚的トークン、人間の音声単位などのモダリティ固有のターゲットを予測するのではなく、
本質的にローカルであるため、data2vec は、からの情報を含む文脈化された潜在表現を予測します。
入力全体。音声認識、画像分類、および
自然言語理解は、新しい最先端技術や、主流のアプローチに匹敵するパフォーマンスを実証します。
モデルとコードは、www.github.com/pytorch/fairseq/tree/master/examples/data2vec.* で入手できます。

このモデルは、[edugp](https://huggingface.co/edugp) および [patrickvonplaten](https://huggingface.co/patrickvonplaten) によって提供されました。
[sayakpaul](https://github.com/sayakpaul) と [Rocketknight1](https://github.com/Rocketknight1) は、TensorFlow のビジョンに Data2Vec を提供しました。

元のコード (NLP および音声用) は、[こちら](https://github.com/pytorch/fairseq/tree/main/examples/data2vec) にあります。
ビジョンの元のコードは [こちら](https://github.com/facebookresearch/data2vec_vision/tree/main/beit) にあります。

## Usage tips

- Data2VecAudio、Data2VecText、および Data2VecVision はすべて、同じ自己教師あり学習方法を使用してトレーニングされています。
- Data2VecAudio の場合、前処理は特徴抽出を含めて [`Wav2Vec2Model`] と同じです。
- Data2VecText の場合、前処理はトークン化を含めて [`RobertaModel`] と同じです。
- Data2VecVision の場合、前処理は特徴抽出を含めて [`BeitModel`] と同じです。

## Resources

Data2Vec の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示される) リソースのリスト。

<PipelineTag pipeline="image-classification"/>

- [`Data2VecVisionForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://cola.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- カスタム データセットで [`TFData2VecVisionForImageClassification`] を微調整するには、[このノートブック](https://colab.research.google.com/github/sayakpaul/TF-2.0-Hacks/blob/master/data2vec_vision_image_classification.ipynb) を参照してください。 ）。

**Data2VecText ドキュメント リソース**
- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [因果言語モデリング タスク ガイド](../tasks/language_modeling)
- [マスク言語モデリング タスク ガイド](../tasks/masked_language_modeling)
- [多肢選択タスク ガイド](../tasks/multiple_choice)

**Data2VecAudio ドキュメント リソース**
- [音声分類タスクガイド](../tasks/audio_classification)
- [自動音声認識タスクガイド](../tasks/asr)

**Data2VecVision ドキュメント リソース**
- [画像分類](../tasks/image_classification)
- [セマンティック セグメンテーション](../tasks/semantic_segmentation)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## Data2VecTextConfig

[[autodoc]] Data2VecTextConfig

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig

## Data2VecVisionConfig

[[autodoc]] Data2VecVisionConfig

<frameworkcontent>
<pt>

## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel
    - forward

## Data2VecAudioForAudioFrameClassification

[[autodoc]] Data2VecAudioForAudioFrameClassification
    - forward

## Data2VecAudioForCTC

[[autodoc]] Data2VecAudioForCTC
    - forward

## Data2VecAudioForSequenceClassification

[[autodoc]] Data2VecAudioForSequenceClassification
    - forward

## Data2VecAudioForXVector

[[autodoc]] Data2VecAudioForXVector
    - forward

## Data2VecTextModel

[[autodoc]] Data2VecTextModel
    - forward

## Data2VecTextForCausalLM

[[autodoc]] Data2VecTextForCausalLM
    - forward

## Data2VecTextForMaskedLM

[[autodoc]] Data2VecTextForMaskedLM
    - forward

## Data2VecTextForSequenceClassification

[[autodoc]] Data2VecTextForSequenceClassification
    - forward

## Data2VecTextForMultipleChoice

[[autodoc]] Data2VecTextForMultipleChoice
    - forward

## Data2VecTextForTokenClassification

[[autodoc]] Data2VecTextForTokenClassification
    - forward

## Data2VecTextForQuestionAnswering

[[autodoc]] Data2VecTextForQuestionAnswering
    - forward

## Data2VecVisionModel

[[autodoc]] Data2VecVisionModel
    - forward

## Data2VecVisionForImageClassification

[[autodoc]] Data2VecVisionForImageClassification
    - forward

## Data2VecVisionForSemanticSegmentation

[[autodoc]] Data2VecVisionForSemanticSegmentation
    - forward

</pt>
<tf>

## TFData2VecVisionModel

[[autodoc]] TFData2VecVisionModel
    - call

## TFData2VecVisionForImageClassification

[[autodoc]] TFData2VecVisionForImageClassification
    - call

## TFData2VecVisionForSemanticSegmentation

[[autodoc]] TFData2VecVisionForSemanticSegmentation
    - call

</tf>
</frameworkcontent>

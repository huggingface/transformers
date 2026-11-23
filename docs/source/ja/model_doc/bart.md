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

# BART

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bart">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bart-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bart-large-mnli">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**免責事項:** 何か奇妙なものを見つけた場合は、[Github 問題](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) を提出し、割り当ててください。
@patrickvonplaten

## Overview

Bart モデルは、[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation、
翻訳と理解](https://huggingface.co/papers/1910.13461) Mike Lewis、Yinhan Liu、Naman Goyal、Marjan 著
ガズビニネジャド、アブデルラフマン・モハメド、オメル・レヴィ、ベス・ストヤノフ、ルーク・ゼトルモイヤー、2019年10月29日。

要約によると、

- Bart は、双方向エンコーダ (BERT など) を備えた標準の seq2seq/機械翻訳アーキテクチャを使用します。
  左から右へのデコーダ (GPT など)。
- 事前トレーニング タスクには、元の文の順序をランダムにシャッフルし、新しい埋め込みスキームが含まれます。
  ここで、テキストの範囲は単一のマスク トークンに置き換えられます。
- BART は、テキスト生成用に微調整した場合に特に効果的ですが、理解タスクにも適しています。それ
  RoBERTa のパフォーマンスを GLUE および SQuAD の同等のトレーニング リソースと同等にし、新たな成果を達成します。
  さまざまな抽象的な対話、質問応答、要約タスクに関する最先端の結果が得られ、成果が得られます。
  ルージュは最大6枚まで。

チップ：

- BART は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左。
- エンコーダーとデコーダーを備えたシーケンスツーシーケンス モデル。エンコーダには破損したバージョンのトークンが供給され、デコーダには元のトークンが供給されます（ただし、通常のトランスフォーマー デコーダと同様に、将来のワードを隠すためのマスクがあります）。次の変換の構成は、エンコーダーの事前トレーニング タスクに適用されます。

  * ランダムなトークンをマスクします (BERT と同様)
  * ランダムなトークンを削除します
  * k 個のトークンのスパンを 1 つのマスク トークンでマスクします (0 トークンのスパンはマスク トークンの挿入です)
  * 文を並べ替えます
  * ドキュメントを回転して特定のトークンから開始するようにします

このモデルは [sshleifer](https://huggingface.co/sshleifer) によって提供されました。著者のコードは [ここ](https://github.com/pytorch/fairseq/tree/master/examples/bart) にあります。

### Examples

- Hugging Face `datasets` を使用して [`BartForConditionalGeneration`] をトレーニングする方法の例
  オブジェクトは、この [フォーラム ディスカッション](https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904) で見つけることができます。
- [抽出されたチェックポイント](https://huggingface.co/models?search=distilbart) は、この [論文](https://huggingface.co/papers/2010.13002) で説明されています。

## Implementation Notes

- Bart はシーケンスの分類に `token_type_ids` を使用しません。 [`BartTokenizer`] を使用するか、
  [`~BartTokenizer.encode`] を使用して適切に分割します。
- [`BartModel`] のフォワードパスは、渡されなかった場合、`decoder_input_ids` を作成します。
  これは、他のモデリング API とは異なります。この機能の一般的な使用例は、マスクの塗りつぶしです。
- モデルの予測は、次の場合に元の実装と同一になるように意図されています。
  `forced_bos_token_id=0`。ただし、これは、渡す文字列が次の場合にのみ機能します。
  [`fairseq.encode`] はスペースで始まります。
- [`~generation.GenerationMixin.generate`] は、次のような条件付き生成タスクに使用する必要があります。
  要約については、その docstring の例を参照してください。
- *facebook/bart-large-cnn* 重みをロードするモデルには `mask_token_id` がないか、実行できません。
  マスクを埋めるタスク。

## Mask Filling

`facebook/bart-base` および `facebook/bart-large` チェックポイントを使用して、マルチトークン マスクを埋めることができます。

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
]
```

## Resources

BART を始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

<PipelineTag pipeline="summarization"/>

- に関するブログ投稿 [分散トレーニング: 🤗 Transformers と Amazon SageMaker を使用した要約のための BART/T5 のトレーニング](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq)。
- 方法に関するノートブック [blurr を使用して fastai で要約するために BART を微調整する](https://colab.research.google.com/github/ohmeow/ohmeow_website/blob/master/posts/2021-05-25-mbart-sequence-classification-with-blurr.ipynb). 🌎 🌎
- 方法に関するノートブック [トレーナー クラスを使用して 2 つの言語で要約するために BART を微調整する](https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb)。 🌎
- [`BartForConditionalGeneration`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)。
- [`TFBartForConditionalGeneration`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)。
- [要約](https://huggingface.co/course/chapter7/5?fw=pt#summarization) 🤗 ハグフェイスコースの章。
- [要約タスクガイド](../tasks/summarization)

<PipelineTag pipeline="fill-mask"/>

- [`BartForConditionalGeneration`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) でサポートされています。
- [`TFBartForConditionalGeneration`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)。
- [`FlaxBartForConditionalGeneration`] は、この [ノートブック]( https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)。
- [マスクされた言語モデリング](https://huggingface.co/course/chapter7/3?fw=pt) 🤗 顔ハグ コースの章。
- [マスクされた言語モデリング タスク ガイド](../tasks/masked_lang_modeling)

<PipelineTag pipeline="translation"/>

- [ヒンディー語から英語への翻訳に Seq2SeqTrainer を使用して mBART を微調整する方法に関するノート](https://colab.research.google.com/github/vasudevgupta7/huggingface-tutorials/blob/main/translation_training.ipynb)。 🌎
- [`BartForConditionalGeneration`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)。
- [`TFBartForConditionalGeneration`] は、この [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)。
- [翻訳タスクガイド](../tasks/translation)

以下も参照してください。
- [テキスト分類タスクガイド(英語版)](../../en/tasks/sequence_classification)
- [質問回答タスク ガイド](../tasks/question_answering)
- [因果言語モデリング タスク ガイド](../tasks/language_modeling)
- [抽出されたチェックポイント](https://huggingface.co/models?search=distilbart) は、この [論文](https://huggingface.co/papers/2010.13002) で説明されています。

## BartConfig

[[autodoc]] BartConfig
    - all

## BartTokenizer

[[autodoc]] BartTokenizer
    - all

## BartTokenizerFast

[[autodoc]] BartTokenizerFast
    - all

## BartModel

[[autodoc]] BartModel
    - forward

## BartForConditionalGeneration

[[autodoc]] BartForConditionalGeneration
    - forward

## BartForSequenceClassification

[[autodoc]] BartForSequenceClassification
    - forward

## BartForQuestionAnswering

[[autodoc]] BartForQuestionAnswering
    - forward

## BartForCausalLM

[[autodoc]] BartForCausalLM
    - forward

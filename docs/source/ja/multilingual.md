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

# 推論のための多言語モデル

[[open-in-colab]]

🤗 Transformers にはいくつかの多言語モデルがあり、それらの推論の使用方法は単一言語モデルとは異なります。ただし、多言語モデルの使用方法がすべて異なるわけではありません。 [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased) などの一部のモデルは、単一言語モデルと同様に使用できます。 このガイドでは、推論のために使用方法が異なる多言語モデルをどのように使うかを示します。

## XLM

XLM には10の異なるチェックポイントがあり、そのうちの1つだけが単一言語です。 残りの9つのモデルチェックポイントは、言語埋め込みを使用するチェックポイントと使用しないチェックポイントの2つのカテゴリに分けることができます。

### 言語の埋め込みがある XLM

次の XLM モデルは、言語の埋め込みを使用して、推論で使用される言語を指定します。

- `FacebookAI/xlm-mlm-ende-1024` (マスク化された言語モデリング、英語-ドイツ語)
- `FacebookAI/xlm-mlm-enfr-1024` (マスク化された言語モデリング、英語-フランス語)
- `FacebookAI/xlm-mlm-enro-1024` (マスク化された言語モデリング、英語-ルーマニア語)
- `FacebookAI/xlm-mlm-xnli15-1024` (マスク化された言語モデリング、XNLI 言語)
- `FacebookAI/xlm-mlm-tlm-xnli15-1024` (マスク化された言語モデリング + 翻訳 + XNLI 言語)
- `FacebookAI/xlm-clm-enfr-1024` (因果言語モデリング、英語-フランス語)
- `FacebookAI/xlm-clm-ende-1024` (因果言語モデリング、英語-ドイツ語)

言語の埋め込みは、モデルに渡される `input_ids` と同じ形状のテンソルとして表されます。 これらのテンソルの値は、使用される言語に依存し、トークナイザーの `lang2id` および `id2lang` 属性によって識別されます。

この例では、`FacebookAI/xlm-clm-enfr-1024` チェックポイントをロードします (因果言語モデリング、英語-フランス語)。

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
```

トークナイザーの `lang2id` 属性は、このモデルの言語とその ID を表示します。

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

次に、入力例を作成します。

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

言語 ID を `en` に設定し、それを使用して言語の埋め込みを定義します。 言語の埋め込みは、英語の言語 ID であるため、`0` で埋められたテンソルです。 このテンソルは `input_ids` と同じサイズにする必要があります。

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # We reshape it to be of size (batch_size, sequence_length)
>>> langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
```

これで、`input_ids` と言語の埋め込みをモデルに渡すことができます。

```py
>>> outputs = model(input_ids, langs=langs)
```

[run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) スクリプトは、`xlm-clm` チェックポイントを使用して、言語が埋め込まれたテキストを生成できます。

### 言語の埋め込みがないXLM

次の XLM モデルは、推論中に言語の埋め込みを必要としません。

- `FacebookAI/xlm-mlm-17-1280` (マスク化された言語モデリング、17の言語)
- `FacebookAI/xlm-mlm-100-1280` (マスク化された言語モデリング、100の言語)

これらのモデルは、以前の XLM チェックポイントとは異なり、一般的な文の表現に使用されます。

## BERT

以下の BERT モデルは、多言語タスクに使用できます。

- `google-bert/bert-base-multilingual-uncased` (マスク化された言語モデリング + 次の文の予測、102の言語)
- `google-bert/bert-base-multilingual-cased` (マスク化された言語モデリング + 次の文の予測、104の言語)

これらのモデルは、推論中に言語の埋め込みを必要としません。 文脈から言語を識別し、それに応じて推測する必要があります。

## XLM-RoBERTa

次の XLM-RoBERTa モデルは、多言語タスクに使用できます。

- `FacebookAI/xlm-roberta-base` (マスク化された言語モデリング、100の言語)
- `FacebookAI/xlm-roberta-large` (マスク化された言語モデリング、100の言語)

XLM-RoBERTa は、100の言語で新しく作成およびクリーニングされた2.5 TB の CommonCrawl データでトレーニングされました。 これは、分類、シーケンスのラベル付け、質問応答などのダウンストリームタスクで、mBERT や XLM などの以前にリリースされた多言語モデルを大幅に改善します。

## M2M100

次の M2M100 モデルは、多言語翻訳に使用できます。

- `facebook/m2m100_418M` (翻訳)
- `facebook/m2m100_1.2B` (翻訳)

この例では、`facebook/m2m100_418M` チェックポイントをロードして、中国語から英語に翻訳します。 トークナイザーでソース言語を設定できます。

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

テキストをトークン化します。

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100 は、最初に生成されたトークンとしてターゲット言語 ID を強制的にターゲット言語に翻訳します。 英語に翻訳するには、`generate` メソッドで `forced_bos_token_id` を `en` に設定します。

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

多言語翻訳には、次の MBart モデルを使用できます。

- `facebook/mbart-large-50-one-to-many-mmt` (One-to-many multilingual machine translation, 50 languages)
- `facebook/mbart-large-50-many-to-many-mmt` (Many-to-many multilingual machine translation, 50 languages)
- `facebook/mbart-large-50-many-to-one-mmt` (Many-to-one multilingual machine translation, 50 languages)
- `facebook/mbart-large-50` (Multilingual translation, 50 languages)
- `facebook/mbart-large-cc25`

この例では、`facebook/mbart-large-50-many-to-many-mmt` チェックポイントをロードして、フィンランド語を英語に翻訳します。トークナイザーでソース言語を設定できます。

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

テキストをトークン化します。

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart は、最初に生成されたトークンとしてターゲット言語 ID を強制的にターゲット言語に翻訳します。 英語に翻訳するには、`generate` メソッドで `forced_bos_token_id` を `en` に設定します。

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id("en_XX"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

`facebook/mbart-large-50-many-to-one-mmt` チェックポイントを使用している場合、最初に生成されたトークンとしてターゲット言語 ID を強制する必要はありません。それ以外の場合、使用方法は同じです。
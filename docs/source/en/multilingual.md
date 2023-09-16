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

# Multilingual models for inference

[[open-in-colab]]

There are several multilingual models in 🤗 Transformers, and their inference usage differs from monolingual models. Not *all* multilingual model usage is different though. Some models, like [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased), can be used just like a monolingual model. This guide will show you how to use multilingual models whose usage differs for inference.

## XLM

XLM has ten different checkpoints, only one of which is monolingual. The nine remaining model checkpoints can be split into two categories: the checkpoints that use language embeddings and those that don't.

### XLM with language embeddings

The following XLM models use language embeddings to specify the language used at inference:

- `xlm-mlm-ende-1024` (Masked language modeling, English-German)
- `xlm-mlm-enfr-1024` (Masked language modeling, English-French)
- `xlm-mlm-enro-1024` (Masked language modeling, English-Romanian)
- `xlm-mlm-xnli15-1024` (Masked language modeling, XNLI languages)
- `xlm-mlm-tlm-xnli15-1024` (Masked language modeling + translation, XNLI languages)
- `xlm-clm-enfr-1024` (Causal language modeling, English-French)
- `xlm-clm-ende-1024` (Causal language modeling, English-German)

Language embeddings are represented as a tensor of the same shape as the `input_ids` passed to the model. The values in these tensors depend on the language used and are identified by the tokenizer's `lang2id` and `id2lang` attributes.

In this example, load the `xlm-clm-enfr-1024` checkpoint (Causal language modeling, English-French):

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
```

The `lang2id` attribute of the tokenizer displays this model's languages and their ids:

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

Next, create an example input:

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

Set the language id as `"en"` and use it to define the language embedding. The language embedding is a tensor filled with `0` since that is the language id for English. This tensor should be the same size as `input_ids`. 

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # We reshape it to be of size (batch_size, sequence_length)
>>> langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
```

Now you can pass the `input_ids` and language embedding to the model:

```py
>>> outputs = model(input_ids, langs=langs)
```

The [run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) script can generate text with language embeddings using the `xlm-clm` checkpoints.

### XLM without language embeddings

The following XLM models do not require language embeddings during inference:

- `xlm-mlm-17-1280` (Masked language modeling, 17 languages)
- `xlm-mlm-100-1280` (Masked language modeling, 100 languages)

These models are used for generic sentence representations, unlike the previous XLM checkpoints.

## BERT

The following BERT models can be used for multilingual tasks:

- `bert-base-multilingual-uncased` (Masked language modeling + Next sentence prediction, 102 languages)
- `bert-base-multilingual-cased` (Masked language modeling + Next sentence prediction, 104 languages)

These models do not require language embeddings during inference. They should identify the language from the
context and infer accordingly.

## XLM-RoBERTa

The following XLM-RoBERTa models can be used for multilingual tasks:

- `xlm-roberta-base` (Masked language modeling, 100 languages)
- `xlm-roberta-large` (Masked language modeling, 100 languages)

XLM-RoBERTa was trained on 2.5TB of newly created and cleaned CommonCrawl data in 100 languages. It provides strong gains over previously released multilingual models like mBERT or XLM on downstream tasks like classification, sequence labeling, and question answering.

## M2M100

The following M2M100 models can be used for multilingual translation:

- `facebook/m2m100_418M` (Translation)
- `facebook/m2m100_1.2B` (Translation)

In this example, load the `facebook/m2m100_418M` checkpoint to translate from Chinese to English. You can set the source language in the tokenizer:

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

Tokenize the text:

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100 forces the target language id as the first generated token to translate to the target language. Set the `forced_bos_token_id` to `en` in the `generate` method to translate to English:

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

The following MBart models can be used for multilingual translation:

- `facebook/mbart-large-50-one-to-many-mmt` (One-to-many multilingual machine translation, 50 languages)
- `facebook/mbart-large-50-many-to-many-mmt` (Many-to-many multilingual machine translation, 50 languages)
- `facebook/mbart-large-50-many-to-one-mmt` (Many-to-one multilingual machine translation, 50 languages)
- `facebook/mbart-large-50` (Multilingual translation, 50 languages)
- `facebook/mbart-large-cc25`

In this example, load the `facebook/mbart-large-50-many-to-many-mmt` checkpoint to translate Finnish to English. You can set the source language in the tokenizer:

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

Tokenize the text:

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart forces the target language id as the first generated token to translate to the target language. Set the `forced_bos_token_id` to `en` in the `generate` method to translate to English:

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

If you are using the `facebook/mbart-large-50-many-to-one-mmt` checkpoint, you don't need to force the target language id as the first generated token otherwise the usage is the same.

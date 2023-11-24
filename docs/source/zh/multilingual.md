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

# 用于推理的多语言模型

[[open-in-colab]]

🤗 Transformers 中有多种多语言模型，它们的推理用法与单语言模型不同。但是，并非*所有*的多语言模型用法都不同。一些模型，例如 [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased) 就可以像单语言模型一样使用。本指南将向您展示如何使用不同用途的多语言模型进行推理。

## XLM

XLM 有十个不同的检查点，其中只有一个是单语言的。剩下的九个检查点可以归为两类：使用语言嵌入的检查点和不使用语言嵌入的检查点。

### 带有语言嵌入的 XLM

以下 XLM 模型使用语言嵌入来指定推理中使用的语言：

- `xlm-mlm-ende-1024` （掩码语言建模，英语-德语）
- `xlm-mlm-enfr-1024` （掩码语言建模，英语-法语）
- `xlm-mlm-enro-1024` （掩码语言建模，英语-罗马尼亚语）
- `xlm-mlm-xnli15-1024` （掩码语言建模，XNLI 数据集语言）
- `xlm-mlm-tlm-xnli15-1024` （掩码语言建模+翻译，XNLI 数据集语言）
- `xlm-clm-enfr-1024` （因果语言建模，英语-法语）
- `xlm-clm-ende-1024` （因果语言建模，英语-德语）

语言嵌入被表示一个张量，其形状与传递给模型的 `input_ids` 相同。这些张量中的值取决于所使用的语言，并由分词器的 `lang2id` 和 `id2lang`  属性识别。

在此示例中，加载 `xlm-clm-enfr-1024` 检查点（因果语言建模，英语-法语）：

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
```

分词器的 `lang2id` 属性显示了该模型的语言及其对应的id：

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

接下来，创建一个示例输入：

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size 为 1
```

将语言 id 设置为 `"en"` 并用其定义语言嵌入。语言嵌入是一个用 `0` 填充的张量，这个张量应该与 `input_ids` 大小相同。

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # 我们将其 reshape 为 (batch_size, sequence_length) 大小
>>> langs = langs.view(1, -1)  # 现在的形状是 [1, sequence_length] (我们的 batch size 为 1)
```

现在，你可以将 `input_ids` 和语言嵌入传递给模型：

```py
>>> outputs = model(input_ids, langs=langs)
```

[run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) 脚本可以使用 `xlm-clm` 检查点生成带有语言嵌入的文本。

### 不带语言嵌入的 XLM

以下 XLM 模型在推理时不需要语言嵌入：

- `xlm-mlm-17-1280` （掩码语言建模，支持 17 种语言）
- `xlm-mlm-100-1280` （掩码语言建模，支持 100 种语言）

与之前的 XLM 检查点不同，这些模型用于通用句子表示。

## BERT

以下 BERT 模型可用于多语言任务：

- `bert-base-multilingual-uncased` （掩码语言建模 + 下一句预测，支持 102 种语言）
- `bert-base-multilingual-cased` （掩码语言建模 + 下一句预测，支持 104 种语言）

这些模型在推理时不需要语言嵌入。它们应该能够从上下文中识别语言并进行相应的推理。

## XLM-RoBERTa

以下 XLM-RoBERTa 模型可用于多语言任务：

- `xlm-roberta-base` （掩码语言建模，支持 100 种语言）
- `xlm-roberta-large` （掩码语言建模，支持 100 种语言）

XLM-RoBERTa 使用 100 种语言的 2.5TB 新创建和清理的 CommonCrawl 数据进行了训练。与之前发布的 mBERT 或 XLM 等多语言模型相比，它在分类、序列标记和问答等下游任务上提供了更强大的优势。

## M2M100

以下 M2M100 模型可用于多语言翻译：

- `facebook/m2m100_418M` （翻译）
- `facebook/m2m100_1.2B` （翻译）

在此示例中，加载 `facebook/m2m100_418M` 检查点以将中文翻译为英文。你可以在分词器中设置源语言：

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

对文本进行分词：

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100 强制将目标语言 id 作为第一个生成的标记，以进行到目标语言的翻译。在 `generate` 方法中将 `forced_bos_token_id` 设置为 `en` 以翻译成英语：

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

以下 MBart 模型可用于多语言翻译：

- `facebook/mbart-large-50-one-to-many-mmt` （一对多多语言机器翻译，支持 50 种语言）
- `facebook/mbart-large-50-many-to-many-mmt` （多对多多语言机器翻译，支持 50 种语言）
- `facebook/mbart-large-50-many-to-one-mmt` （多对一多语言机器翻译，支持 50 种语言）
- `facebook/mbart-large-50` （多语言翻译，支持 50 种语言）
- `facebook/mbart-large-cc25`

在此示例中，加载  `facebook/mbart-large-50-many-to-many-mmt` 检查点以将芬兰语翻译为英语。 你可以在分词器中设置源语言：

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

对文本进行分词：

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart 强制将目标语言 id 作为第一个生成的标记，以进行到目标语言的翻译。在 `generate` 方法中将 `forced_bos_token_id` 设置为 `en` 以翻译成英语：

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

如果你使用的是 `facebook/mbart-large-50-many-to-one-mmt` 检查点，则无需强制目标语言 id 作为第一个生成的令牌，否则用法是相同的。

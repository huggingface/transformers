<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；在不违反许可证的情况下，您可能不会使用此文件。您可以在下面获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按原样分发的软件在许可证下分发基础上，“按原样” BASIS，无论是明示还是暗示，不带任何形式的担保或条件。请参阅许可证特定语言的权限和限制。
⚠️请注意，此文件采用 Markdown 格式，但包含特定于我们的 doc-builder（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 用于推理的多语言模型

[[在 Colab 中打开]]

在🤗 Transformers 中有几个多语言模型，它们的推理用法与单语模型不同。但并非 *所有* 多语言模型的用法都不同。某些模型，如 [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)，可以像单语模型一样使用。本指南将向您展示如何使用用于推理的用法与其他多语言模型不同的多语言模型。
## XLM

XLM 有十个不同的检查点，其中只有一个是单语的。其余九个模型检查点可以分为两类：使用语言嵌入的检查点和不使用语言嵌入的检查点。

### 使用语言嵌入的 XLM

以下 XLM 模型在推理时使用语言嵌入来指定使用的语言：
- `xlm-mlm-ende-1024`（掩码语言建模，英德）
- `xlm-mlm-enfr-1024`（掩码语言建模，英法）
- `xlm-mlm-enro-1024`（掩码语言建模，英罗）
- `xlm-mlm-xnli15-1024`（掩码语言建模，XNLI 系列语言）
- `xlm-mlm-tlm-xnli15-1024`（掩码语言建模+翻译，XNLI 系列语言）
- `xlm-clm-enfr-1024`（因果语言建模，英法）
- `xlm-clm-ende-1024`（因果语言建模，英德）

语言嵌入表示为与传递给模型的 `input_ids` 形状相同的张量。这些张量中的值取决于使用的语言，并由标记器的 `lang2id` 和 `id2lang` 属性标识。

在此示例中，加载 `xlm-clm-enfr-1024` 检查点（因果语言建模，英法）：
```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
```

标记器的 `lang2id` 属性显示了该模型的语言及其 ID：
```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

接下来，创建一个示例输入：
```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

将语言 ID 设置为 `"en"` 并使用它来定义语言嵌入。语言嵌入是一个填充为 `0` 的张量，因为这是英语的语言 ID。该张量应与 `input_ids` 的大小相同。
```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # We reshape it to be of size (batch_size, sequence_length)
>>> langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
```

现在，您可以将 `input_ids` 和语言嵌入传递给模型：
```py
>>> outputs = model(input_ids, langs=langs)
```

[run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) 脚本可以使用 `xlm-clm` 检查点生成具有语言嵌入的文本。

### 不使用语言嵌入的 XLM

以下 XLM 模型在推理时不需要语言嵌入：
- `xlm-mlm-17-1280`（掩码语言建模，17 种语言）
- `xlm-mlm-100-1280`（掩码语言建模，100 种语言）
这些模型用于通用句子表示，与之前的 XLM 检查点不同。
## BERT
以下 BERT 模型可用于多语言任务：
- `bert-base-multilingual-uncased`（掩码语言建模+下一句预测，102 种语言）
- `bert-base-multilingual-cased`（掩码语言建模+下一句预测，104 种语言）
在推理过程中，这些模型不需要语言嵌入。它们应该从上下文中识别语言并相应地进行推理。
## XLM-RoBERTa

以下 XLM-RoBERTa 模型可用于多语言任务：
- `xlm-roberta-base`（掩码语言建模，100 种语言）
- `xlm-roberta-large`（掩码语言建模，100 种语言）

XLM-RoBERTa 在 100 种语言的新创建和清理的 CommonCrawl 数据上进行了 2.5TB 的训练。它在分类、序列标注和问答等下游任务上相比先前发布的多语言模型（如 mBERT 或 XLM）取得了很大的提升。

## M2M100

以下 M2M100 模型可用于多语言翻译：
- `facebook/m2m100_418M`（翻译）
- `facebook/m2m100_1.2B`（翻译）

在此示例中，加载 `facebook/m2m100_418M` 检查点以将中文翻译为英文。您可以在标记器中设置源语言：

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

M2M100 将目标语言 ID 强制为要翻译到的目标语言的第一个生成的标记。在 `generate` 方法中将 `forced_bos_token_id` 设置为 `en` 以进行英语翻译：
```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

以下 MBart 模型可用于多语言翻译：
- `facebook/mbart-large-50-one-to-many-mmt`（一对多多语言机器翻译，50 种语言）
- `facebook/mbart-large-50-many-to-many-mmt`（多对多多语言机器翻译，50 种语言）
- `facebook/mbart-large-50-many-to-one-mmt`（多对一多语言机器翻译，50 种语言）- `facebook/mbart-large-50`（多语言翻译，50 种语言）
- `facebook/mbart-large-cc25`
在此示例中，加载 `facebook/mbart-large-50-many-to-many-mmt` 检查点以将芬兰语翻译为英语。您可以在标记器中设置源语言：
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

MBart 将目标语言 ID 强制为要翻译到的目标语言的第一个生成的标记。在 `generate` 方法中将 `forced_bos_token_id` 设置为 `en` 以进行英语翻译：
```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id("en_XX"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

如果您使用的是 `facebook/mbart-large-50-many-to-one-mmt` 检查点，则不需要将目标语言 ID 强制为第一个生成的标记，否则使用方法相同。
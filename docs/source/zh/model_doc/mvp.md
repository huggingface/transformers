<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）的规定，除非符合许可证的要求，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取特定语言下的权限和限制。请注意，该文件虽然是 Markdown 格式，但包含特定的语法，用于我们的文档生成器（类似于 MDX），在您的 Markdown 查看器中可能无法正确显示。
⚠️ 请注意，此文件为 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。注意：
-->
# MVP

## 概述

MVP 模型是由 Tianyi Tang，Junyi Li，Wayne Xin Zhao 和 Ji-Rong Wen 在 [MVP: 多任务监督预训练自然语言生成](https://arxiv.org/abs/2206.12131) 中提出的。

根据摘要，

- MVP 遵循标准的 Transformer 编码器-解码器架构。- MVP 使用有标签的数据进行监督预训练。- MVP 还具有特定任务的软提示，以激发模型在执行某个任务时的能力。- MVP 专为自然语言生成而设计，可适用于广泛的生成任务，包括但不限于摘要，数据到文本生成，开放式对话系统，故事生成，问答，问题生成，任务导向对话系统，常识生成，转述生成，文本风格转换和文本简化。

我们的模型还可以适用于自然语言理解任务，如序列分类和（抽取式）问答。

提示：

- 我们在 [这里](https://huggingface.co/models?filter=mvp) 发布了一系列模型，包括 MVP，具有特定任务提示的 MVP 和多任务预训练变体。
- 如果您想使用没有提示的模型（标准 Transformer），您可以通过 `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp')` 加载它。- 如果您想使用具有特定任务提示的模型，例如摘要，您可以通过 `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp-summarization')` 加载它。
- 我们的模型支持使用“Prefix-tuning”进行轻量级提示调整，方法是 `set_lightweight_tuning()`。
此模型由 [Tianyi Tang](https://huggingface.co/StevenTang) 贡献。

详细信息和说明可在 [此处](https://github.com/RUCAIBox/MVP) 找到。

## 示例

对于摘要，使用 MVP 和具有摘要特定提示的 MVP 的示例。

```python
>>> from transformers import MvpTokenizer, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_prompt = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp-summarization")

>>> inputs = tokenizer(
...     "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["Why You Shouldn't Quit Your Job"]

>>> generated_ids = model_with_prompt.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["Don't do it if these are your reasons"]
```

对于数据到文本生成，使用 MVP 和多任务预训练变体的示例。

```python
>>> from transformers import MvpTokenizerFast, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizerFast.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_mtl = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")

>>> inputs = tokenizer(
...     "Describe the following data: Iron Man | instance of | Superhero [SEP] Stan Lee | creator | Iron Man",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['Stan Lee created the character of Iron Man, a fictional superhero appearing in American comic']

>>> generated_ids = model_with_mtl.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['Iron Man is a fictional superhero appearing in American comic books published by Marvel Comics.']
```

对于轻量级调整，即固定模型仅调整提示，您可以加载具有随机初始化提示或具有特定任务提示的 MVP。

我们的代码还支持使用 BART 进行“Prefix-tuning”，遵循 [原始论文](https://arxiv.org/abs/2101.00190)。

```python
>>> from transformers import MvpForConditionalGeneration

>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp", use_prompt=True)
>>> # the number of trainable parameters (full tuning)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
468116832

>>> # lightweight tuning with randomly initialized prompts
>>> model.set_lightweight_tuning()
>>> # the number of trainable parameters (lightweight tuning)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
61823328

>>> # lightweight tuning with task-specific prompts
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")
>>> model.set_lightweight_tuning()
>>> # original lightweight Prefix-tuning
>>> model = MvpForConditionalGeneration.from_pretrained("facebook/bart-large", use_prompt=True)
>>> model.set_lightweight_tuning()
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## MvpConfig

[[autodoc]] MvpConfig

## MvpTokenizer

[[autodoc]] MvpTokenizer

## MvpTokenizerFast

[[autodoc]] MvpTokenizerFast

## MvpModel

[[autodoc]] MvpModel
    - forward

## MvpForConditionalGeneration

[[autodoc]] MvpForConditionalGeneration
    - forward

## MvpForSequenceClassification

[[autodoc]] MvpForSequenceClassification
    - forward

## MvpForQuestionAnswering

[[autodoc]] MvpForQuestionAnswering
    - forward

## MvpForCausalLM

[[autodoc]] MvpForCausalLM
    - forward

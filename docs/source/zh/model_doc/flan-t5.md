<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获得许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按现状提供”方式分发，不附带任何明示或暗示的保证或条件。请参阅许可证以了解许可的具体语言、权限和限制。⚠️ 请注意，此文件是 Markdown 格式，但包含了我们文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确渲染。特定语言、权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了我们文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确渲染。特定语言、权限和限制。
-->
# FLAN-T5

## 概述

FLAN-T5 在论文 [扩展指令微调语言模型](https://arxiv.org/pdf/2210.11416.pdf) 中发布——它是 T5 的增强版本，已在多个任务的混合数据上进行了微调。
可以直接使用 FLAN-T5 的权重而无需微调模型：
```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Pour a cup of bolognese into a large bowl and add the pasta']
```

FLAN-T5 包含与 T5 版本 1.1 相同的改进（有关模型改进的详细信息，请参阅 [此处](https://huggingface.co/docs/transformers/model_doc/t5v1.1)）。
Google 发布了以下变种：
- [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
- [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
- [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)。

可以参考 [T5 的文档页面](t5) 获取所有提示、代码示例和笔记本。还可以查看 FLAN-T5 模型卡片以获取有关模型的训练和评估的更多详细信息。

原始检查点可以在 [此处](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) 找到。
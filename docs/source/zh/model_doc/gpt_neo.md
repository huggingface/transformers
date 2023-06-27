<!--版权所有 2021 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”基础分发的，不提供任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确呈现。
-->
# GPT Neo

## 概述

GPTNeo 模型是由 Sid Black、Stella Biderman、Leo Gao、Phil Wang 和 Connor Leahy 在 [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) 代码仓库中发布的。它是基于 [Pile](https://pile.eleuther.ai/) 数据集进行训练的类似 GPT2 的因果语言模型。[Pile](https://pile.eleuther.ai/) dataset.

架构与 GPT2 类似，不同之处在于 GPT Neo 在每一层中使用局部注意力，窗口大小为 256 个标记。
此模型由 [valhalla](https://huggingface.co/valhalla) 贡献。

### 生成

`generate()` 方法可用于使用 GPT Neo 模型生成文本。
```python
>>> from transformers import GPTNeoForCausalLM, GPT2Tokenizer

>>> model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
>>> tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```


## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)- [因果语言建模任务指南](../tasks/language_modeling)

## GPTNeoConfig

[[autodoc]] GPTNeoConfig

## GPTNeoModel

[[autodoc]] GPTNeoModel
    - forward

## GPTNeoForCausalLM

[[autodoc]] GPTNeoForCausalLM
    - forward

## GPTNeoForQuestionAnswering

[[autodoc]] GPTNeoForQuestionAnswering
    - forward

## GPTNeoForSequenceClassification

[[autodoc]] GPTNeoForSequenceClassification
    - forward

## GPTNeoForTokenClassification

[[autodoc]] GPTNeoForTokenClassification
    - forward

## FlaxGPTNeoModel

[[autodoc]] FlaxGPTNeoModel
    - __call__

## FlaxGPTNeoForCausalLM

[[autodoc]] FlaxGPTNeoForCausalLM
    - __call__

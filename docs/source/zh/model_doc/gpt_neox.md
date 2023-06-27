<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“原样”分发，不附带任何形式的保证或条件。有关详细信息，请参阅许可证特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。
-->
# GPT-NeoX

## 概述

我们介绍了 GPT-NeoX-20B，这是一个 200 亿参数的自回归语言模型，它是在 Pile 上训练的，其权重将通过宽松的许可证免费向公众提供。据我们所知，这是公开可用的最大的密集自回归模型。在此工作中，我们描述了 GPT-NeoX-20B 的架构和训练，并评估了它在一系列语言理解、数学和基于知识的任务上的性能。我们发现，与相似规模的 GPT-3 和 FairSeq 模型相比，GPT-NeoX-20B 是一个特别强大的少样本推理器，在进行五次测试时，它的性能提高得更多。

我们在 [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) 上开源了训练和评估代码以及模型权重。该模型的开发由 Sid Black、Stella Biderman 和 Eric Hallahan 领导，
并在 [CoreWeave](https://www.coreweave.com/) 的大力支持下进行了训练。generous the support of [CoreWeave](https://www.coreweave.com/).


GPT-NeoX-20B 是使用 fp16 进行训练的，因此建议按以下方式初始化模型：

```python
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()
```

GPT-NeoX-20B 与 GPT-J-6B 和 GPT-Neo 使用的分词器 (Tokenizer)不同。新的分词器 (Tokenizer)为空白字符分配了额外的标记，使得该模型更适用于某些任务，如代码生成。

### 生成

可以使用 `generate()` 方法使用 GPT Neo 模型生成文本。

```python
>>> from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

>>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
>>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

>>> prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."

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

- [因果语言建模任务指南](../tasks/language_modeling)

## GPTNeoXConfig

[[autodoc]] GPTNeoXConfig

## GPTNeoXTokenizerFast

[[autodoc]] GPTNeoXTokenizerFast

## GPTNeoXModel

[[autodoc]] GPTNeoXModel
    - forward

## GPTNeoXForCausalLM

[[autodoc]] GPTNeoXForCausalLM
    - forward

## GPTNeoXForQuestionAnswering

[[autodoc]] GPTNeoXForQuestionAnswering
    - forward

## GPTNeoXForSequenceClassification

[[autodoc]] GPTNeoXForSequenceClassification
    - forward

## GPTNeoXForTokenClassification

[[autodoc]] GPTNeoXForTokenClassification
    - forward

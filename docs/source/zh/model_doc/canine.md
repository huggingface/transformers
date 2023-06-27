<!--版权所有 2021 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版本（“许可证”）许可；除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不附带任何形式的担保或条件，明示或暗示。请参阅许可证了解特定语言下的权限和限制。
⚠️请注意，此文件使用 Markdown 编写，但包含特定于我们文档构建器（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# CANINE

## 概述

CANINE 模型是由 Jonathan H. Clark、Dan Garrette、Iulia Turc 和 John Wieting 在 [CANINE: Pre-training an Efficient Tokenization-Free Encoder for LanguageRepresentation](https://arxiv.org/abs/2103.06874) 提出的。

这是一篇最早的不使用显式分词步骤（如字节对编码（BPE）、WordPiece 或 SentencePiece）的 Transformer 进行训练的论文之一。相反，该模型直接在 Unicode 字符级别进行训练。字符级别的训练不可避免地会导致较长的序列长度，CANINE 通过一种高效的降采样策略解决了这个问题，然后应用一个深度 Transformer 编码器。该论文的摘要如下：


*虽然基于流水线的 NLP 系统大部分已被端到端的神经建模所取代，但几乎所有常用的模型仍然需要显式的分词步骤。尽管最近基于数据衍生的子词词典的分词方法比手工设计的分词器 (Tokenizer)更加灵活，但这些技术并不适用于所有语言，并且使用任何固定的词汇表可能会限制模型的适应能力。在本文中，我们介绍了 CANINE，它是一个在不使用显式的分词或词汇表的情况下直接对字符序列进行编码的神经编码器，并提出了一种预训练策略，该策略可以直接使用字符或可选地使用子词作为软性归纳偏差。为了有效和高效地使用更精细的输入，CANINE 结合了降采样（减少输入序列长度）和深度 Transformer 堆栈（编码上下文）。尽管模型参数减少了 28%，CANINE 在具有挑战性的多语言基准测试 TyDi QA 上的 F1 值比可比的 mBERT 模型高出 2.8%。*

提示：

- CANINE 在内部使用了 3 个 Transformer 编码器：2 个“浅层”编码器（仅包含一个  层）和 1 个“深层”编码器（正常的 BERT 编码器）。首先，使用“浅层”编码器对字符嵌入进行  上下文化处理，使用局部注意力。接下来，在降采样之后应用“深层”编码器。最后，在上采样之后  使用“浅层”编码器创建最终的字符嵌入。有关上采样和下采样的详细信息，请参阅论文。  

- CANINE 默认使用最大序列长度 2048 个字符。可以使用 [`CanineTokenizer`]  对文本进行处理。- 分类可以通过在特殊的 [CLS] 标记的最终隐藏状态上放置一个线性层来完成  （该标记具有预定义的 Unicode 代码点）。然而，对于标记分类任务，需要将下采样的标记序列  再次上采样以匹配原始字符序列的长度（为 2048 个字符）。有关此操作的详细信息  请参阅论文。

- 模型：
  - [google/canine-c](https://huggingface.co/google/canine-c)：基于自回归字符损失    预训练，12 层，768 隐藏层，12 个头，121M 参数（大小约为 500MB）。  
  - [google/canine-s](https://huggingface.co/google/canine-s)：基于子词损失预训练，12 层，    768 隐藏层，12 个头，121M 参数（大小约为 500MB）。

此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在 [这里](https://github.com/google-research/language/tree/master/language/canine) 找到。

### 示例

CANINE 适用于原始字符，因此可以在没有分词器 (Tokenizer)的情况下使用：

```python
>>> from transformers import CanineModel
>>> import torch

>>> model = CanineModel.from_pretrained("google/canine-c")  # model pre-trained with autoregressive character loss

>>> text = "hello world"
>>> # use Python's built-in ord() function to turn each character into its unicode code point id
>>> input_ids = torch.tensor([[ord(char) for char in text]])

>>> outputs = model(input_ids)  # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

但是，对于批量推理和训练，建议使用分词器 (Tokenizer)（以将所有序列填充/截断为相同长度）：
```python
>>> from transformers import CanineTokenizer, CanineModel

>>> model = CanineModel.from_pretrained("google/canine-c")
>>> tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

>>> inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
>>> encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

>>> outputs = model(**encoding)  # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [多项选择任务指南](../tasks/multiple_choice)

## CANINE specific outputs

[[autodoc]] models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineConfig

[[autodoc]] CanineConfig

## CanineTokenizer

[[autodoc]] CanineTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences

## CanineModel

[[autodoc]] CanineModel
    - forward

## CanineForSequenceClassification

[[autodoc]] CanineForSequenceClassification
    - forward

## CanineForMultipleChoice

[[autodoc]] CanineForMultipleChoice
    - forward

## CanineForTokenClassification

[[autodoc]] CanineForTokenClassification
    - forward

## CanineForQuestionAnswering

[[autodoc]] CanineForQuestionAnswering
    - forward

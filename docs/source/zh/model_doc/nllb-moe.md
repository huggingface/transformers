<!--版权2023年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证 2.0 版（“许可证”）授权，除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，该文件是 Markdown 格式，但包含我们的文档构建工具（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确渲染。
-->
# NLLB-MOE

## 概述

NLLB 模型在 Marta R. Costa-juss à、James Cross、Onur Ç elebi 等人的论文 [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) 中提出。Maha Elbayad、Kenneth Heafield、Kevin Heffernan、Elahe Kalbassi、Janice Lam、Daniel Licht、Jean Maillard、Anna Sun、Skyler Wang、Guillaume Wenzek、Al Youngblood、Bapi Akula 等人参与了该论文的撰写。Loic Barrault、Gabriel Mejia Gonzalez、Prangthip Hansanti、John Hoffman、Semarley Jarrett、Kaushik Ram Sadagopan、Dirk Rowe、Shannon Spruit、Chau Tran、Pierre Andrews 等人也参与了该论文的撰写。Necip Fazil Ayan、Shruti Bhosale、Sergey Edunov、Angela Fan、Cynthia Gao、Vedanuj Goswami、Francisco Guzm á n、Philipp Koehn、Alexandre Mourachko、Christophe Ropers、Safiyyah Saleem、Holger Schwenk 和 Jeff Wang。Safiyyah Saleem, Holger Schwenk, and Jeff Wang.

该论文的摘要如下：

*机器翻译以消除全球范围内的语言障碍为目标，已经成为当今人工智能研究的重点。然而，这些努力主要集中在一小部分语言上，大多数资源较少的语言被忽视。要克服这个问题，我们需要破解 200 种语言的障碍，并确保安全、高质量的结果，同时考虑伦理问题。在《没有语言被遗漏》中，我们通过与母语讲者进行初探性访谈，来了解低资源语言翻译支持的需求。然后，我们创建了针对低资源语言的数据集和模型，旨在缩小低资源语言和高资源语言之间的性能差距。具体而言，我们基于 Sparsely Gated Mixture of Experts 开发了一种基于条件计算的模型，该模型使用专为低资源语言量身定制的新颖有效的数据挖掘技术获取数据。我们提出了多种架构和训练改进措施来对抗训练过度拟合的问题，并使用人工翻译的基准数据集 Flores-200 评估了超过 40,000 个不同的翻译方向的性能，同时结合了人工评估和覆盖 Flores-200 中所有语言的新颖毒性基准来评估翻译的安全性。我们的模型相对于先前的最新技术改进了 44%的 BLEU，为实现通用翻译系统奠定了重要的基础。* 

提示：

- M2M100ForConditionalGeneration 是 NLLB 和 NLLB MoE 的基础模型- NLLB-MoE 与 NLLB 模型非常相似，但其前馈层基于 SwitchTransformers 实现。

- 分词器 (Tokenizer)与 NLLB 模型相同。
此模型由 [Arthur Zucker](https://huggingface.co/ArtZucker) 贡献。原始代码可在 [此处](https://github.com/facebookresearch/fairseq) 找到。

## 与 SwitchTransformers 的实现差异最大的区别在于令牌的路由方式。NLLB-MoE 使用 `top-2-gate`，这意味着对于每个输入，只有两个具有最高预测概率的专家被选择，其他专家被忽略。而在 `SwitchTransformers` 中，仅计算前 1 个概率，这意味着令牌被转发的概率较小。此外，如果一个令牌没有路由到任何专家，`SwitchTransformers` 仍会添加其未修改的隐藏状态（类似于残差连接），而在 `NLLB` 的 top-2 路由机制中，这些令牌被掩码。states (kind of like a residual connection) while they are masked in `NLLB`'s top-2 routing mechanism. 

## 使用 NLLB-MoE 

进行生成可用的检查点需要大约 350GB 的存储空间。如果您的计算机内存不足，请确保使用 `accelerate` 库。
在生成目标文本时，将 `forced_bos_token_id` 设置为目标语言的 ID。以下是使用 *facebook/nllb-200-distilled-600M* 模型从英语翻译成法语的示例。请注意，我们在法语的 BCP-47 代码中使用了 `fra_Latn`。请参阅 [此处](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
以获取 Flores 200 数据集中所有 BCP-47 的列表。

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")

>>> article = "Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage."
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=50
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer que la société avait commencé lorsque sa sonnette n'était pas audible depuis son magasin dans son garage."
```

### 从非英语语言生成

默认情况下，将英语（`eng_Latn`）设置为翻译源语言。如果您想从其他语言翻译，请在分词器 (Tokenizer)初始化的 `src_lang` 关键字参数中指定 BCP-47 代码。

以下示例演示了从罗马尼亚语翻译为德语的情况：

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b", src_lang="ron_Latn")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```

## 文档资源

- [翻译任务指南](../tasks/translation)- [摘要任务指南](../tasks/summarization)

## NllbMoeConfig

[[autodoc]] NllbMoeConfig

## NllbMoeTop2Router

[[autodoc]] NllbMoeTop2Router
    - route_tokens
    - forward

## NllbMoeSparseMLP

[[autodoc]] NllbMoeSparseMLP
    - forward

## NllbMoeModel

[[autodoc]] NllbMoeModel
    - forward

## NllbMoeForConditionalGeneration

[[autodoc]] NllbMoeForConditionalGeneration
    - forward

<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）进行许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。请注意，此文件为 Markdown 格式，但包含我们 doc-builder 的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法正常
渲染。
-->

# BARThez

## 概述

BARThez 模型是由 Moussa Kamal Eddine、Antoine J.-P. Tixier 和 Michalis Vazirgiannis 于 2020 年 10 月 23 日在 [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) 中提出的。

该论文的摘要如下:

*归纳迁移学习，通过自监督学习实现，已经在整个自然语言处理（NLP）领域中取得了巨大的成功，BERT 和 BART 等模型在无数自然语言理解任务中树立了新的技术水平。

尽管有一些显著的例外，但大多数可用的模型和研究都是针对英语进行的。

在这项工作中，我们介绍了 BARThez，这是第一个针对法语的 BART 模型 （据我们所知）。

BARThez 使用来自过去研究的非常大型的法语单语语料库进行预训练，我们对其进行了适应以适应 BART 的扰动方案。

（如 CamemBERT 和 FlauBERT）不同，BARThez 特别适用于生成任务，因为它的编码器和解码器都经过了预训练。

除了 FLUE 基准测试的判别任务外，我们还在一个新的总结数据集 OrangeSum 上评估了 BARThezits 并与本文一起发布。

我们还继续在 BARThez 的语料库上预训练一个已经预训练的多语言 BART，我们称之为 mBARTHez，结果表明这种模型明显优于纯粹的 BARThez，并且与 CamemBERT 和 FlauBERT 持平或超过它们。

*
该模型由 [moussakam](https://huggingface.co/moussakam) 贡献。作者的代码可以在 [这里](https://github.com/moussaKam/BARThez) 找到。

### 示例


- BARThez 可以像 BART 一样在序列到序列任务上进行微调，请参考:  [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md)。

## BarthezTokenizer

[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast

[[autodoc]] BarthezTokenizerFast

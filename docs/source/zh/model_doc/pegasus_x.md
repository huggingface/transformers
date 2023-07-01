<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用本文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何明示或暗示的担保或条件。请查看许可证以获得特定语言下的权限和限制。请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能无法正确地在您的 Markdown 查看器中呈现。
⚠️请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能无法正确地在您的 Markdown 查看器中呈现。请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能无法正确地在您的 Markdown 查看器中呈现。
-->
# PEGASUS-X

## 概述

PEGASUS-X 模型是由 Jason Phang、Yao Zhao 和 Peter J. Liu 在 [Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347) 中提出的。

PEGASUS-X（PEGASUS 扩展）通过额外的长输入预训练和在编码器中使用交错的块本地注意力和全局标记来扩展 PEGASUS 模型以用于长输入摘要。

论文中的摘要如下：

*尽管大型预训练 Transformer 模型在处理自然语言任务方面表现出色，但处理长序列输入仍然是一个重大挑战。其中一个任务是长输入摘要，其中输入超过大多数预训练模型的最大输入上下文。通过一系列广泛的实验，我们研究了哪些模型架构变化和预训练范式可以最有效地适应预训练 Transformer 用于长输入摘要。我们发现，具有全局编码器标记的交错块本地 Transformer 在性能和效率之间取得了良好的平衡，并且在长序列上进行额外的预训练可以显着提高下游摘要性能。基于我们的发现，我们引入了 PEGASUS-X，这是 PEGASUS 模型的扩展，通过额外的长输入预训练来处理长达 16K 标记的输入。PEGASUS-X 在长输入摘要任务上表现出与更大模型相当的强大性能，同时增加了少量的额外参数，并且不需要模型并行训练。*

提示：

* PEGASUS-X 使用与 PEGASUS 相同的分词器。
此模型由 [zphang](<https://huggingface.co/zphang>) 贡献。原始代码可在 [此处](https://github.com/google-research/pegasus) 找到。

## 文档资源

- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## PegasusXConfig

[[autodoc]] PegasusXConfig


## PegasusXModel

[[autodoc]] PegasusXModel
    - forward


## PegasusXForConditionalGeneration

[[autodoc]] PegasusXForConditionalGeneration
    - forward

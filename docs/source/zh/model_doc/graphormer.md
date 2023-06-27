<!--版权所有 2022 年 HuggingFace 团队和 Microsoft。保留所有权利。-->
根据 MIT 许可证获得许可；除非符合许可证的要求，否则您不得使用此文件。许可证。
除非适用法律要求或书面同意，根据许可证分发的软件是按 "原样" 分发的，不附带任何形式的明示或暗示的保证或条件。请参阅许可证中的特定语言以获取权限和限制。特定语言以获取权限和限制。特定语言以获取权限和限制。
⚠️ 请注意，此文件是使用 Markdown 编写的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确呈现。呈现。
-->
# Graphormer

## 概述

Graphormer 模型是由 [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234) 提出的，作者是 Chengxuan Ying、Tianle Cai、Shengjie Luo、Shuxin Zheng、Guolin Ke、Di He、Yanming Shen 和 Tie-Yan Liu。它是一个图转换器模型，通过在预处理和整理过程中生成感兴趣的嵌入和特征，然后使用修改后的注意力机制，在图上进行计算，而不是在文本序列上进行计算。
以下是论文的摘要：

*Transformer 架构已成为许多领域的主要选择，如自然语言处理和计算机视觉。然而，与主流的图神经网络变体相比，在流行的图级预测排行榜上，它的性能还没有达到竞争力。因此，如何使 Transformer 在图表示学习上表现良好仍然是一个谜。在本文中，我们通过提出 Graphormer 来解决这个谜题，它建立在标准的 Transformer 架构之上，并且在广泛的图表示学习任务中取得了出色的结果，尤其是在最近的 OGB 大规模挑战中。我们在利用 Transformer 在图中的关键洞察是将图的结构信息有效地编码到模型中。为此，我们提出了几种简单而有效的结构编码方法，以帮助 Graphormer 更好地建模图结构化数据。此外，我们对 Graphormer 的表达能力进行了数学刻画，并展示了通过我们的图结构信息编码方法，许多流行的图神经网络变体可以作为 Graphormer 的特殊情况。*

小贴士：
此模型对于大型图（超过 100 个节点/边）效果不佳，因为会导致内存溢出。您可以减小批次大小、增加内存或减小 algos_graphormer.pyx 中的 `UNREACHABLE_NODE_DISTANCE` 参数，但很难超过 700 个节点/边。

此模型不使用分词器 (Tokenizer)，而是在训练过程中使用特殊的整理器。

此模型由 [clefourrier](https://huggingface.co/clefourrier) 提供。原始代码可在 [此处](https://github.com/microsoft/Graphormer) 找到。


## GraphormerConfig

[[autodoc]] GraphormerConfig


## GraphormerModel

[[autodoc]] GraphormerModel
    - forward


## GraphormerForGraphClassification

[[autodoc]] GraphormerForGraphClassification
    - forward
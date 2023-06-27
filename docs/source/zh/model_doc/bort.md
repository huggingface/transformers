<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可，除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件是按照“原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# BORT

## 概述

BORT 模型是由 Adrian de Wynter 和 Daniel J. Perry 在《Optimal Subarchitecture Extraction for BERT》中提出的。这是 BERT 的一种最佳子架构参数集合，作者称之为“Bort”。该论文的摘要如下：
*我们通过应用神经架构搜索算法的最新突破，从 Devlin 等人（2018）的 BERT 架构中提取出一种最佳子架构参数集合。

这种最佳子集，我们称之为“Bort”，明显更小，其有效大小（不计算嵌入层）为原始 BERT-large 架构的 5.5%，是净大小的 16%。Bort 的预训练时间为 288 个 GPU 小时，相当于最高性能的 BERT 参数化架构变体 RoBERTa-large（Liu 等人，2019）预训练时间的 1.2%，以及在相同硬件上训练 BERT-large 所需的世界记录的 33%。它在 CPU 上也快了 7.9 倍，性能也优于架构的其他压缩变体和一些非压缩变体：绝对上，在多个公共自然语言理解（NLU）基准测试中，相对于 BERT-large，性能提升在 0.3%到 31%之间。

提示：


- BORT 的模型架构基于 BERT，因此可以参考 [BERT 的文档页面](bert) 了解模型的 API 以及使用示例。
- BORT 使用 RoBERTa 分词器 (Tokenizer)而不是 BERT 分词器 (Tokenizer)，因此可以参考 [RoBERTa 的文档页面](roberta) 了解分词器 (Tokenizer)的 API 以及使用示例。


- BORT 需要一种特定的微调算法，称为 [Agora](https://adewynter.github.io/notes/bort_algorithms_and_applications.html#fine-tuning-with-algebraic-topology)，  不幸的是，该算法尚未开源。如果有人尝试实现该算法以使 BORT 微调工作，对社区将非常有用。  
。
该模型由 [stefan-it](https://huggingface.co/stefan-it) 贡献。原始代码可在 [此处](https://github.com/alexa/bort/) 找到。
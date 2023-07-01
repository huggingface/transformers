<!--版权2020年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在许可证处获得许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或默示的保证或条件。请参阅许可证以了解特定语言下权限和限制的详细信息。⚠️请注意，此文件是 Markdown 格式的，但其中包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中适当地渲染。
-->
# CPM

## 概述

CPM 模型是由 Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin,
Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen,
Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.在 [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413) 中提出的。

论文中的摘要如下所示：

*预训练语言模型（PLMs）已被证明对各种下游 NLP 任务有益。最近，拥有 1750 亿参数和 570GB 训练数据的 GPT-3 由于能够进行少量甚至零-shot 学习而引起了很大关注。然而，将 GPT-3 应用于解决中文 NLP 任务仍然具有挑战性，因为 GPT-3 的训练语料库主要是英文，参数不是公开可用的。在本技术报告中，我们发布了具有大规模中文训练数据的中文预训练语言模型（CPM）。据我们所知，CPM 是拥有 26 亿参数和 100GB 中文训练数据的最大中文预训练语言模型，可以促进几个下游的中文 NLP 任务，如对话、文章生成、填空测试和语言理解。广泛的实验证明，CPM 在许多 NLP 任务的少量甚至零-shot 学习设置下都能取得良好的性能。*

此模型由 [canwenxu](https://huggingface.co/canwenxu) 贡献。

原始实现可以在这里找到：https://github.com/TsinghuaAI/CPM-Generate

注意：我们这里只有一个分词器 (Tokenizer)，因为模型架构与 GPT-2 相同。

## CpmTokenizer

[[autodoc]] CpmTokenizer

## CpmTokenizerFast

[[autodoc]] CpmTokenizerFast

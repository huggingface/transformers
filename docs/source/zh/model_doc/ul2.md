<!--版权所有 2022 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件。请参阅许可证以了解具体语言规定的权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。-->

# UL2

T5 模型是由 Yi Tay，Mostafa Dehghani，Vinh Q. Tran，Xavier Garcia，Dara Bahri，Tal Schuster，Huaixiu Steven Zheng，Neil Houlsby，Donald Metzler 在 [Unifying Language Learning Paradigms](https://arxiv.org/pdf/2205.05131v1.pdf) 中提出的。

论文的摘要如下：
*现有的预训练模型通常针对特定类别的问题。迄今为止，关于合适的架构和预训练设置仍然没有共识。本文提出了一个统一的框架，用于预训练在数据集和设置上都具有普适性的模型。我们首先将架构原型与预训练目标解开，这两个概念通常混淆。接下来，我们提出了自我监督在 NLP 中的广义和统一视角，并展示了如何将不同的预训练目标转化为彼此，并且插值不同的目标可以是有效的。然后，我们提出了混合去噪器（MoD），这是一种将多种预训练范式结合在一起的预训练目标。我们还引入了一种模式切换的概念，其中下游微调与特定的预训练方案相关联。我们进行了广泛的剖析实验，比较了多种预训练目标，并发现我们的方法通过在多个不同的设置中优于 T5 和/或 GPT 类似模型推动了 Pareto 前沿。最后，通过将模型扩展到 200 亿个参数，我们在 50 个成熟的有监督 NLP 任务上实现了 SOTA 性能，包括语言生成（自动和人工评估）、语言理解、文本分类、问答、常识推理、长文本推理、结构化知识基础和信息检索。我们的模型在上下文学习方面也取得了强大的结果，在零样本 SuperGLUE 上超过了 1750 亿个 GPT-3，并且在单次摘要中将 T5-XXL 的性能提升了三倍。*

提示：

- UL2 是一个编码器-解码器模型，它在一系列下游任务上进行了混合去噪函数的预训练和微调。

- UL2 具有与 [T5v1.1](t5v1.1) 相同的架构，但使用的是 Gated-SiLU 激活函数而不是 Gated-GELU。

- 作者在 [此处](https://huggingface.co/google/ul2) 发布了一个架构的检查点

原始代码可以在 [此处](https://github.com/google-research/google-research/tree/master/ul2) 找到。

此模型由 [DanielHesslow](https://huggingface.co/Seledorn) 贡献。
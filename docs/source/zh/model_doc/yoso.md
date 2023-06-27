<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”），您只能在符合以下条件的情况下使用此文件许可证。您可以在以下网址获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS，不提供任何形式的担保或条件，无论是明示还是暗示。请参阅许可证特定语言下权限和限制的具体规定。
⚠️请注意，此文件是 Markdown 格式的，但包含我们 doc-builder 的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。
-->
# YOSO

## 概述

YOSO 模型在 [You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://arxiv.org/abs/2111.09714) 中提出作者为 Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh。YOSO 通过基于局部敏感哈希（LSH）的伯努利抽样方案来近似标准的 softmax 自注意力原理上，所有的伯努利随机变量都可以通过单次哈希采样单个哈希。

论文摘要如下：

*基于 Transformer 的模型在自然语言处理（NLP）中被广泛使用。Transformer 模型的核心是自注意力机制，它捕捉输入序列中令牌对之间的交互，并且与序列长度呈二次关系。在本文中，我们展示了一种基于局部敏感哈希（LSH）的伯努利采样注意机制，将这类模型的二次复杂度降低为线性。我们通过将自注意力视为与伯努利随机变量相关联的单个令牌的总和，可以一次性通过单个哈希采样（尽管在实践中，此数字可能是一个小常数）。这导致了一种有效的采样方案来估计自注意力，该方案依赖于特定的修改 LSH（以在 GPU 架构上进行部署）。我们在标准的 512 序列上使用 GLUE 基准评估了我们的算法长度，在这里我们看到与标准的预训练 Transformer 相比有利的性能。在 Long Range Arena（LRA）基准测试中，用于评估长序列性能，我们的方法实现了与 softmax 自注意力一致的结果，但具有可观的加速和节省内存，并且通常优于其他高效的自注意力方法。我们的代码在此 https URL 中提供*

提示：

- YOSO 注意力算法通过自定义 CUDA 内核实现，CUDA C++编写的可以在 GPU 上并行执行多次的函数。内核提供 `fast_hash` 函数，使用快速哈达玛变换近似查询和键的随机投影。使用这些哈希码，`lsh_cumulation` 函数通过基于 LSH 的伯努利采样近似自注意力。hash codes, the `lsh_cumulation` function approximates self-attention via LSH-based Bernoulli sampling.
- 要使用自定义内核，用户应设置 `config.use_expectation = False`。

为确保内核成功编译，用户必须安装正确版本的 PyTorch 和 cudatoolkit。默认情况下，`config.use_expectation = True`，使用 YOSO-E 不需要编译 CUDA 内核。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yoso_architecture.jpg"alt="drawing" width="600"/> 

<small> YOSO 注意力算法。摘自 <a href="https://arxiv.org/abs/2111.09714"> 原始论文 </a>。</small>

此模型由 [novice03](https://huggingface.co/novice03) 贡献。原始代码在 [这里](https://github.com/mlpen/YOSO) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [遮蔽语言模型任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## YosoConfig

[[autodoc]] YosoConfig


## YosoModel

[[autodoc]] YosoModel
    - forward


## YosoForMaskedLM

[[autodoc]] YosoForMaskedLM
    - forward


## YosoForSequenceClassification

[[autodoc]] YosoForSequenceClassification
    - forward

## YosoForMultipleChoice

[[autodoc]] YosoForMultipleChoice
    - forward


## YosoForTokenClassification

[[autodoc]] YosoForTokenClassification
    - forward


## YosoForQuestionAnswering

[[autodoc]] YosoForQuestionAnswering
    - forward
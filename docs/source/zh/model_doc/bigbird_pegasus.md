<!--版权所有2021年HuggingFace团队。保留所有权利。
- ->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不能使用此文件。您可以在以下位置获取许可证副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不附带任何形式的担保或条件。请参阅许可证了解特定语言下的许可证和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似 MDX）的特定语法，您的 Markdown 查看器可能无法正确呈现。
-->
# BigBirdPegasus

## 概览

BigBird 模型是由 Zaheer, Manzil 和 Guruganesh, Guru 和 Dubey, Kumar Avinava 和 Ainslie, Joshua 和 Alberti, Chris 和 Ontanon, Santiago 和 Pham, Philip 和 Ravula, Anirudh 和 Wang, Qifan 和 Yang, Li 等人在 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) 中提出的。BigBird 是一种基于稀疏注意力的变形器，它将基于 Transformer 的模型（如 BERT）扩展到更长的序列。除了稀疏注意力外，BigBird 还对输入序列应用全局注意力和随机注意力。从理论上讲，已经证明了应用稀疏、全局和随机注意力可以近似全局注意力，同时在处理更长的序列时具有更高的计算效率。由于处理更长的上下文的能力，BigBird 在各种长文档 NLP 任务（如问答和摘要）中相比 BERT 或 RoBERTa 表现出更好的性能。
论文中的摘要如下所示：
The abstract from the paper is the following:

*基于 Transformer 的模型（如 BERT）一直是 NLP 领域最成功的深度学习模型之一。不幸的是，它们的一个核心限制是由于全局注意力机制导致对于序列长度（主要是内存方面）的二次依赖性。为了解决这个问题，我们提出了 BigBird，一种稀疏注意力机制，将这种二次依赖性减少到线性。我们展示了 BigBird 是序列函数的通用逼近器，并且是图灵完备的，从而保留了二次依赖性和全局注意力模型的这些特性。在此过程中，我们的理论分析揭示了在稀疏注意力机制的一部分中具有 O（1）全局标记（如 CLS）的一些好处。提出的稀疏注意力可以处理比以前更长的序列长度，使用相似硬件可达到的长度的 8 倍。由于处理更长的上下文的能力，BigBird 在各种 NLP 任务（如问答和摘要）中有着显著的改进。我们还提出了
对基因组数据的新应用。*

提示：

- 有关 BigBird 的注意力机制的详细解释，请参阅 [此博文](https://huggingface.co/blog/big-bird)。
- BigBird 有两种实现方式：**original_full** 和 **block_sparse**。对于序列长度 < 1024，建议使用  **original_full**，因为使用**block_sparse**注意力没有任何好处。
-  代码当前使用 3 个块和 2 个全局块的窗口大小。
-  序列长度必须是块大小的整数倍。
- 当前实现仅支持**ITC**。

- 当前实现不支持**num_random_blocks = 0**。
-  BigBirdPegasus 使用[PegasusTokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pegasus/tokenization_pegasus.py)。
-  BigBird 是一个带有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。

可以在 [此处](https://github.com/google-research/bigbird) 找到原始代码。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)- [翻译任务指南](../tasks/translation)- [摘要任务指南](../tasks/summarization)

## BigBirdPegasusConfig

[[autodoc]] BigBirdPegasusConfig
    - all

## BigBirdPegasusModel

[[autodoc]] BigBirdPegasusModel
    - forward

## BigBirdPegasusForConditionalGeneration

[[autodoc]] BigBirdPegasusForConditionalGeneration
    - forward

## BigBirdPegasusForSequenceClassification

[[autodoc]] BigBirdPegasusForSequenceClassification
    - forward

## BigBirdPegasusForQuestionAnswering

[[autodoc]] BigBirdPegasusForQuestionAnswering
    - forward

## BigBirdPegasusForCausalLM

[[autodoc]] BigBirdPegasusForCausalLM
    - forward

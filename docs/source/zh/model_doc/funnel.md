<!--版权 2020 年 The HuggingFace 团队。版权所有。
根据 Apache 许可证版本 2.0（“许可证”）授权；您除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件将按一个“按原样”基础分发，不提供任何明示或暗示的保证或条件。请参阅许可证特定语言下的权限和限制。
⚠️请注意，此文件为 Markdown 格式，但包含我们的 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# Funnel Transformer

<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=funnel"> <img alt="Models" src="https://img.shields.io/badge/All_model_pages-funnel-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/funnel-transformer-small"> <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

## 概述

Funnel Transformer 模型是在论文 [Funnel-Transformer：过滤顺序冗余以进行高效语言处理](https://arxiv.org/abs/2006.03236) 中提出的。它是一个双向变压器模型，类似于 BERT，但在每个块的层之后进行汇聚操作，有点像传统的计算机视觉中的卷积神经网络（CNN）。

论文中的摘要如下所示：

*随着语言预训练的成功，开发更高效的、可扩展性良好的体系结构，以更低的成本利用丰富的无标签数据，是非常希望的。为了提高效率，我们研究了维护完整的令牌级表示中经常被忽视的冗余，特别是对于只需要序列的单一向量表示的任务。基于这个直觉，我们提出了 Funnel-Transformer，它逐渐将隐藏状态序列压缩为较短的序列，从而减少计算量。更重要的是，通过将长度缩减的节省 FLOPs 重新投入构建更深或更宽的模型中，我们进一步改进了模型的容量。此外，为了执行常见的预训练目标所要求的令牌级预测，Funnel-Transformer 能够从缩减的隐藏序列中为每个令牌恢复一个深度表示，通过解码器实现。经验证明，具有可比较或更少 FLOPs 的 Funnel-Transformer 在各种序列级预测任务上（包括文本分类、语言理解和阅读理解）优于标准 Transformer。*

提示：

- 由于 Funnel Transformer 使用汇聚操作，隐藏状态的序列长度在每个块的层之后会发生变化。这样，它们的长度被除以 2，从而加速了计算下一个隐藏状态的过程。因此，基础模型的最终序列长度是原始序列长度的四分之一。该模型可以直接用于只需要句子摘要的任务（例如序列分类或多项选择）。对于其他任务，使用完整模型；该完整模型具有一个解码器，可以将最终的隐藏状态上采样到与输入相同的序列长度。
- 对于分类等任务，这不是一个问题，但对于像掩码语言建模或令牌分类这样的任务，我们需要一个与原始输入序列长度相同的隐藏状态。在这些情况下，最终的隐藏状态被上采样到输入序列长度，并经过两个额外的层。这就是为什么每个检查点都有两个版本的原因。带有“-base”后缀的版本仅包含三个块，而不带该后缀的版本则包含三个块和上采样头和其额外的层。
- Funnel Transformer 的检查点都有完整版本和基础版本。前者应该用于 [`FunnelModel`]，[`FunnelForPreTraining`]，[`FunnelForMaskedLM`]，[`FunnelForTokenClassification`] 和 [`FunnelForQuestionAnswering`]。后者应该用于 [`FunnelBaseModel`]，[`FunnelForSequenceClassification`] 和 [`FunnelForMultipleChoice`]。  [`FunnelForMultipleChoice`].

此模型由 [sgugger](https://huggingface.co/sgugger) 贡献。原始代码可在 [此处](https://github.com/laiguokun/Funnel-Transformer) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)- [标记分类任务指南](../tasks/token_classification)- [问答任务指南](../tasks/question_answering)- [掩码语言建模任务指南](../tasks/masked_language_modeling)- [多项选择任务指南](../tasks/multiple_choice)

## FunnelConfig

[[autodoc]] FunnelConfig

## FunnelTokenizer

[[autodoc]] FunnelTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FunnelTokenizerFast

[[autodoc]] FunnelTokenizerFast

## Funnel specific outputs

[[autodoc]] models.funnel.modeling_funnel.FunnelForPreTrainingOutput

[[autodoc]] models.funnel.modeling_tf_funnel.TFFunnelForPreTrainingOutput

## FunnelBaseModel

[[autodoc]] FunnelBaseModel
    - forward

## FunnelModel

[[autodoc]] FunnelModel
    - forward

## FunnelModelForPreTraining

[[autodoc]] FunnelForPreTraining
    - forward

## FunnelForMaskedLM

[[autodoc]] FunnelForMaskedLM
    - forward

## FunnelForSequenceClassification

[[autodoc]] FunnelForSequenceClassification
    - forward

## FunnelForMultipleChoice

[[autodoc]] FunnelForMultipleChoice
    - forward

## FunnelForTokenClassification

[[autodoc]] FunnelForTokenClassification
    - forward

## FunnelForQuestionAnswering

[[autodoc]] FunnelForQuestionAnswering
    - forward

## TFFunnelBaseModel

[[autodoc]] TFFunnelBaseModel
    - call

## TFFunnelModel

[[autodoc]] TFFunnelModel
    - call

## TFFunnelModelForPreTraining

[[autodoc]] TFFunnelForPreTraining
    - call

## TFFunnelForMaskedLM

[[autodoc]] TFFunnelForMaskedLM
    - call

## TFFunnelForSequenceClassification

[[autodoc]] TFFunnelForSequenceClassification
    - call

## TFFunnelForMultipleChoice

[[autodoc]] TFFunnelForMultipleChoice
    - call

## TFFunnelForTokenClassification

[[autodoc]] TFFunnelForTokenClassification
    - call

## TFFunnelForQuestionAnswering

[[autodoc]] TFFunnelForQuestionAnswering
    - call

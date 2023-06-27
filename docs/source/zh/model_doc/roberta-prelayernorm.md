<!--版权所有2022年The HuggingFace团队。保留所有权利。-->
根据 Apache 许可证，第 2.0 版（“许可证”）获得的许可；除非符合许可证要求，否则您不得使用此文件。您可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式的，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法正确渲染在您的 Markdown 查看器中。-->



# RoBERTa-PreLayerNorm

## 概述

RoBERTa-PreLayerNorm 模型是由 Myle Ott、Sergey Edunov、Alexei Baevski、Angela Fan、Sam Gross、Nathan Ng、David Grangier 和 Michael Auli 在 [fairseq: A Fast, Extensible Toolkit for Sequence Modeling](https://arxiv.org/abs/1904.01038) 中提出的。它与在 [fairseq](https://fairseq.readthedocs.io/) 中使用 `--encoder-normalize-before` 标志完全相同。

论文摘要如下：

*fairseq 是一个开源的序列建模工具包，允许研究人员和开发人员为翻译、摘要、语言建模和其他文本生成任务训练自定义模型。该工具包基于 PyTorch，并支持跨多个 GPU 和机器的分布式训练。我们还支持在现代 GPU 上进行快速的混合精度训练和推理。*


提示：
- 该实现与 [Roberta](roberta) 相同，只是在使用 _Add and Norm_ 时使用 _Norm and Add_。 _Add_ 和 _Norm_ 是指 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中描述的加法和层归一化。- 这与在 [fairseq](https://fairseq.readthedocs.io/) 中使用 `--encoder-normalize-before` 标志完全相同。

此模型由 [andreasmaden](https://huggingface.co/andreasmaden) 贡献。原始代码可以在 [此处](https://github.com/princeton-nlp/DinkyTrain) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)
## RobertaPreLayerNormConfig

[[autodoc]] RobertaPreLayerNormConfig

## RobertaPreLayerNormModel

[[autodoc]] RobertaPreLayerNormModel
    - forward

## RobertaPreLayerNormForCausalLM

[[autodoc]] RobertaPreLayerNormForCausalLM
    - forward

## RobertaPreLayerNormForMaskedLM

[[autodoc]] RobertaPreLayerNormForMaskedLM
    - forward

## RobertaPreLayerNormForSequenceClassification

[[autodoc]] RobertaPreLayerNormForSequenceClassification
    - forward

## RobertaPreLayerNormForMultipleChoice

[[autodoc]] RobertaPreLayerNormForMultipleChoice
    - forward

## RobertaPreLayerNormForTokenClassification

[[autodoc]] RobertaPreLayerNormForTokenClassification
    - forward

## RobertaPreLayerNormForQuestionAnswering

[[autodoc]] RobertaPreLayerNormForQuestionAnswering
    - forward

## TFRobertaPreLayerNormModel

[[autodoc]] TFRobertaPreLayerNormModel
    - call

## TFRobertaPreLayerNormForCausalLM

[[autodoc]] TFRobertaPreLayerNormForCausalLM
    - call

## TFRobertaPreLayerNormForMaskedLM

[[autodoc]] TFRobertaPreLayerNormForMaskedLM
    - call

## TFRobertaPreLayerNormForSequenceClassification

[[autodoc]] TFRobertaPreLayerNormForSequenceClassification
    - call

## TFRobertaPreLayerNormForMultipleChoice

[[autodoc]] TFRobertaPreLayerNormForMultipleChoice
    - call

## TFRobertaPreLayerNormForTokenClassification

[[autodoc]] TFRobertaPreLayerNormForTokenClassification
    - call

## TFRobertaPreLayerNormForQuestionAnswering

[[autodoc]] TFRobertaPreLayerNormForQuestionAnswering
    - call

## FlaxRobertaPreLayerNormModel

[[autodoc]] FlaxRobertaPreLayerNormModel
    - __call__

## FlaxRobertaPreLayerNormForCausalLM

[[autodoc]] FlaxRobertaPreLayerNormForCausalLM
    - __call__

## FlaxRobertaPreLayerNormForMaskedLM

[[autodoc]] FlaxRobertaPreLayerNormForMaskedLM
    - __call__

## FlaxRobertaPreLayerNormForSequenceClassification

[[autodoc]] FlaxRobertaPreLayerNormForSequenceClassification
    - __call__

## FlaxRobertaPreLayerNormForMultipleChoice

[[autodoc]] FlaxRobertaPreLayerNormForMultipleChoice
    - __call__

## FlaxRobertaPreLayerNormForTokenClassification

[[autodoc]] FlaxRobertaPreLayerNormForTokenClassification
    - __call__

## FlaxRobertaPreLayerNormForQuestionAnswering

[[autodoc]] FlaxRobertaPreLayerNormForQuestionAnswering
    - __call__

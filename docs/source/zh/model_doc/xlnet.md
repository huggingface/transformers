<!--版权所有2020年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何明示或暗示的担保或条件。请参阅许可证的特定语言规定权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含我们 doc-builder（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法
正确呈现渲染。
-->
# XLNet

<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=xlnet"> <img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlnet-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/xlnet-base-cased"> <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

## 概述

XLNet 模型是由 Zhilin Yang、Zihang Dai、Yiming Yang、Jaime Carbonell、Ruslan Salakhutdinov 和 Quoc V. Le 在 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) 中提出的。

XLNet 是使用自回归方法进行预训练的 Transformer-XL 模型的扩展，以通过对输入序列分解顺序的所有排列最大化期望似然来学习双向上下文顺序。

论文中的摘要如下所示：


*通过建模双向上下文的能力，基于去噪自编码的 BERT 预训练方法在超越基于自回归语言模型的预训练方法方面实现了更好的性能。然而，依靠使用掩码对输入进行损坏，BERT 忽略了掩码位置之间的依赖关系，并且在预训练和微调之间存在差异。考虑到这些优缺点，我们提出了 XLNet，一种广义的自回归预训练方法，它（1）通过对分解顺序的所有排列最大化期望似然来实现双向上下文的学习，以及（2）通过其自回归的形式克服了 BERT 的局限性。此外，XLNet 还将 Transformer-XL 这一最先进的自回归模型的思想整合到了预训练中。根据可比的实验设置，XLNet 在 20 个任务中，常常以较大的较大的边缘，包括问题回答、自然语言推理、情感分析和文档排序等方面，优于 BERT。
- 可以使用 `perm_mask` 输入来控制特定的注意模式，无论是在训练还是测试时。*

Tips:

- 由于在各种分解顺序上训练一个完全的自回归模型的困难，XLNet 只使用选择的 `target_mapping` 输入作为目标进行预训练使用了一个子集的输出标记。- 要在顺序解码中使用 XLNet（即在完全双向设置中不使用），请使用 `perm_mask` 和 `target_mapping` 输入来控制注意范围和输出（参见  *examples/pytorch/text-generation/run_generation.py* 中的示例）。- XLNet 是为数不多的没有序列长度限制的模型之一。- XLNet 不是传统的自回归模型，而是基于传统自回归模型的训练策略构建的。它对句子中的标记进行排列，然后允许模型使用最后 n 个标记来预测第 n+1 个标记。由于这是使用掩码完成的，因此实际上将句子按正确的顺序输入模型，但是 XLNet 并不是将前 n 个标记掩盖为 n+1，而是使用一个隐藏给定排列中前面标记的掩码。

该模型由 [thomwolf](https://huggingface.co/thomwolf) 贡献。原始代码可以在 [此处](https://github.com/zihangdai/xlnet/) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## XLNetConfig

[[autodoc]] XLNetConfig

## XLNetTokenizer

[[autodoc]] XLNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLNetTokenizerFast

[[autodoc]] XLNetTokenizerFast

## XLNet specific outputs

[[autodoc]] models.xlnet.modeling_xlnet.XLNetModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetModelOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetLMHeadModelOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForQuestionAnsweringSimpleOutput

## XLNetModel

[[autodoc]] XLNetModel
    - forward

## XLNetLMHeadModel

[[autodoc]] XLNetLMHeadModel
    - forward

## XLNetForSequenceClassification

[[autodoc]] XLNetForSequenceClassification
    - forward

## XLNetForMultipleChoice

[[autodoc]] XLNetForMultipleChoice
    - forward

## XLNetForTokenClassification

[[autodoc]] XLNetForTokenClassification
    - forward

## XLNetForQuestionAnsweringSimple

[[autodoc]] XLNetForQuestionAnsweringSimple
    - forward

## XLNetForQuestionAnswering

[[autodoc]] XLNetForQuestionAnswering
    - forward

## TFXLNetModel

[[autodoc]] TFXLNetModel
    - call

## TFXLNetLMHeadModel

[[autodoc]] TFXLNetLMHeadModel
    - call

## TFXLNetForSequenceClassification

[[autodoc]] TFXLNetForSequenceClassification
    - call

## TFLNetForMultipleChoice

[[autodoc]] TFXLNetForMultipleChoice
    - call

## TFXLNetForTokenClassification

[[autodoc]] TFXLNetForTokenClassification
    - call

## TFXLNetForQuestionAnsweringSimple

[[autodoc]] TFXLNetForQuestionAnsweringSimple
    - call

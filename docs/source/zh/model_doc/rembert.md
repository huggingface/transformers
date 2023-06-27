<!--版权所有2020年The HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”），您不得使用此文件，除非符合许可证的规定。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，依据许可证分发的软件是基于“按原样”（AS IS）的基础，无论明示或暗示，不提供任何形式的担保或条件。有关特定语言的权限和限制，请参阅许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# RemBERT

## 概述

RemBERT 模型是由 Hyung Won Chung、Thibault F é vry、Henry Tsai 和 Melvin Johnson 在 [Rethinking Embedding Coupling in Pre-trained Language Models](https://arxiv.org/abs/2010.12821) 中提出的。

该论文的摘要如下：

*我们重新评估了在最先进的预训练语言模型中共享输入和输出嵌入权重的标准做法。我们表明，解耦嵌入提供了更大的建模灵活性，使我们能够在多语言模型的输入嵌入中显着提高参数分配的效率。通过重新分配 Transformer 层中的输入嵌入参数，我们在微调过程中使用相同数量的参数显著提高了标准自然语言理解任务的性能。我们还表明，即使在预训练后丢弃输出嵌入，为模型分配额外的容量也能带来好处。我们的分析表明，更大的输出嵌入防止模型的最后几层过度专门化预训练任务，并鼓励 Transformer 表示更加通用和可转移至其他任务和语言。利用这些发现，我们能够在不增加微调阶段的参数数量的情况下，在 XTREME 基准上训练出性能强大的模型。* .*

提示：

对于微调，可以将 RemBERT 视为 mBERT 的更大版本，其具有类似 ALBERT 的分解输入嵌入层。在预训练中，嵌入是不绑定的，与 BERT 相反，这使得输入嵌入较小（在微调期间保持不变），而输出嵌入较大（在微调期间丢弃）。分词器 (Tokenizer)也类似于 Albert 而不是 BERT。

## 文档资源
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)
## RemBertConfig

[[autodoc]] RemBertConfig

## RemBertTokenizer

[[autodoc]] RemBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertTokenizerFast

[[autodoc]] RemBertTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertModel

[[autodoc]] RemBertModel
    - forward

## RemBertForCausalLM

[[autodoc]] RemBertForCausalLM
    - forward

## RemBertForMaskedLM

[[autodoc]] RemBertForMaskedLM
    - forward

## RemBertForSequenceClassification

[[autodoc]] RemBertForSequenceClassification
    - forward

## RemBertForMultipleChoice

[[autodoc]] RemBertForMultipleChoice
    - forward

## RemBertForTokenClassification

[[autodoc]] RemBertForTokenClassification
    - forward

## RemBertForQuestionAnswering

[[autodoc]] RemBertForQuestionAnswering
    - forward

## TFRemBertModel

[[autodoc]] TFRemBertModel
    - call

## TFRemBertForMaskedLM

[[autodoc]] TFRemBertForMaskedLM
    - call

## TFRemBertForCausalLM

[[autodoc]] TFRemBertForCausalLM
    - call

## TFRemBertForSequenceClassification

[[autodoc]] TFRemBertForSequenceClassification
    - call

## TFRemBertForMultipleChoice

[[autodoc]] TFRemBertForMultipleChoice
    - call

## TFRemBertForTokenClassification

[[autodoc]] TFRemBertForTokenClassification
    - call

## TFRemBertForQuestionAnswering

[[autodoc]] TFRemBertForQuestionAnswering
    - call

<!--版权所有 2020 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件在许可证下分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。注意：此文件采用 Markdown 格式，但包含特定语法以供我们的文档构建器（类似于 MDX）使用，可能在 Markdown 查看器中无法正确显示。
⚠️请注意，此文件采用 Markdown 格式，但包含特定语法以供我们的文档构建器（类似于 MDX）使用，可能在 Markdown 查看器中无法正确显示。
-->
# DeBERTa-v2

## 概述

DeBERTa 模型是由Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen在 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) 一文中提出的。它基于 2018 年发布的 GoogleBERT 模型和 2019 年发布的 Facebook 的 RoBERTa 模型。

它基于 RoBERTa 模型，并引入了解缠结的注意力机制和增强的掩码解码器训练，训练数据量仅为 RoBERTa 模型的一半。论文中的摘要如下：


*最近在预训练神经语言模型方面取得的进展，显著提高了许多自然语言处理（NLP）任务的性能。在本文中，我们提出了一种新的模型架构 DeBERTa（Decoding-enhanced BERTwith disentangled attention），通过两种新颖技术改进了 BERT 和 RoBERTa 模型。第一种技术是解缠结的注意力机制，其中每个单词由两个向量表示，分别编码其内容和位置，并且单词之间的注意力权重是使用其内容和相对位置的解缠结矩阵计算的。其次，使用增强的掩码解码器来替换输出 softmax 层，以预测模型预训练中的掩码标记。我们证明了这两种技术显著提高了模型预训练的效率和下游任务的性能。与 RoBERTa-Large 相比，使用一半训练数据训练的 DeBERTa 模型在各种 NLP 任务上表现更好，在 MNLI 上提高了 0.9%（90.2% vs. 91.1%），在 SQuAD v2.0 上提高了 2.3%（88.4% vs. 90.7%）和 RACE 上提高了 3.6%（83.2% vs. 86.8%）。DeBERTa 的代码和预训练模型将在 https://github.com/microsoft/DeBERTa 上公开。* 


以下信息直接可见于 [原始实现存储库](https://github.com/microsoft/DeBERTa)。DeBERTa v2 是 DeBERTa 模型的第二个版本。它包含 1.5B 模型，用于 SuperGLUE 单模型提交，并取得了 89.9 的分数，而人类基准是 89.8。

您可以在作者的 [博客](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/) 上找到有关此提交的更多详细信息。

v2 中的新功能：

- **词汇表** v2 中的分词器 (Tokenizer)已更改为使用从训练数据中构建的大小为 128K 的新词汇表。分词器 (Tokenizer)不再是基于 GPT2 的，而是  基于 [sentencepiece](https://github.com/google/sentencepiece) 的分词器 (Tokenizer)。  

- **nGiE（nGram Induced Input Encoding）** DeBERTa-v2 模型在第一个 transformer 层旁边使用了一个额外的卷积层，以更好地学习输入标记的局部依赖性。  

- **在注意力层中共享位置投影矩阵和内容投影矩阵** 根据以前的实验，这可以节省参数而不影响性能。  

- **应用桶来编码相对位置** DeBERTa-v2 模型使用对数桶来编码相对位置，类似于 T5。 

- **900M 模型和 1.5B 模型** 提供了两个额外的模型大小：900M 和 1.5B，这显著提高了下游任务的性能。  

此模型由 [DeBERTa](https://huggingface.co/DeBERTa) 贡献。此模型的 TF 2.0 实现由 [kamalkraj](https://huggingface.co/kamalkraj) 贡献。

原始代码可在 [此处](https://github.com/microsoft/DeBERTa) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)
## DebertaV2Tokenizer

[[autodoc]] DebertaV2Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaV2TokenizerFast

[[autodoc]] DebertaV2TokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

## DebertaV2Model

[[autodoc]] DebertaV2Model
    - forward

## DebertaV2PreTrainedModel

[[autodoc]] DebertaV2PreTrainedModel
    - forward

## DebertaV2ForMaskedLM

[[autodoc]] DebertaV2ForMaskedLM
    - forward

## DebertaV2ForSequenceClassification

[[autodoc]] DebertaV2ForSequenceClassification
    - forward

## DebertaV2ForTokenClassification

[[autodoc]] DebertaV2ForTokenClassification
    - forward

## DebertaV2ForQuestionAnswering

[[autodoc]] DebertaV2ForQuestionAnswering
    - forward

## DebertaV2ForMultipleChoice

[[autodoc]] DebertaV2ForMultipleChoice
    - forward

## TFDebertaV2Model

[[autodoc]] TFDebertaV2Model
    - call

## TFDebertaV2PreTrainedModel

[[autodoc]] TFDebertaV2PreTrainedModel
    - call

## TFDebertaV2ForMaskedLM

[[autodoc]] TFDebertaV2ForMaskedLM
    - call

## TFDebertaV2ForSequenceClassification

[[autodoc]] TFDebertaV2ForSequenceClassification
    - call

## TFDebertaV2ForTokenClassification

[[autodoc]] TFDebertaV2ForTokenClassification
    - call

## TFDebertaV2ForQuestionAnswering

[[autodoc]] TFDebertaV2ForQuestionAnswering
    - call

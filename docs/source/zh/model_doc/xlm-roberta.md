<!--版权所有 2020 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证，第 2 版（“许可证”）进行许可；您除非符合许可证的规定，否则不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，不附带任何形式的担保或条件。请参阅许可证中的具体语言，以了解权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确呈现。-->



# XLM-RoBERTa

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlm-roberta">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlm--roberta-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlm-roberta-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

XLM-RoBERTa 模型是由 Alexis Conneau、Kartikay Khandelwal、Naman Goyal、Vishrav Chaudhary、Guillaume Wenzek、Francisco Guzm á n、Edouard Grave、Myle Ott、Luke Zettlemoyer 和 Veselin Stoyanov 在《规模上的无监督跨语言表示学习》中提出的。它基于 Facebook 于 2019 年发布的 RoBERTa 模型。它是一个大型的多语言语言模型，使用了 2.5TB 的经过筛选的 CommonCrawl 数据进行训练。

论文的摘要如下所示：

*本文表明，大规模预训练多语言语言模型能够在广泛的跨语言迁移任务中取得显著的性能提升。我们在一百种语言上训练了基于 Transformer 的遮蔽语言模型，使用了超过两千兆字节的经过筛选的 CommonCrawl 数据。我们的模型被称为 XLM-R，在各种跨语言基准测试中明显优于多语言 BERT（mBERT），包括 XNLI 平均准确性提高了 13.8 ％，MLQA 平均 F1 分数提高了 12.3 ％，NER 平均 F1 分数提高了 2.1 ％。XLM-R 在低资源语言上表现尤为出色，相比之前的 XLM 模型，其在斯瓦希里语的 XNLI 准确性提高了 11.8 ％，乌尔都语提高了 9.2 ％。我们还对实现这些增益所需的关键因素进行了详细的实证评估，包括（1）正向迁移和容量稀释之间的权衡，以及（2）高资源和低资源语言的性能。最后，我们首次展示了在不牺牲每种语言性能的情况下进行多语言建模的可能性；XLM-R 在 GLUE 和 XNLI 基准测试中与强大的单语模型竞争力十足。我们将公开提供 XLM-R 的代码、数据和模型。* 

提示：

- XLM-RoBERTa 是一个在 100 种不同语言上训练的多语言模型。与某些 XLM 多语言模型不同，它不需要 `lang` 张量来确定使用的语言，并且应该能够从输入 id 中确定正确的语言。- 使用 RoBERTa 的技巧进行 XLM 方法，但不使用翻译语言建模目标。它只使用来自一种语言的句子进行遮蔽语言建模。  
- 此实现与 RoBERTa 相同。有关用法示例以及输入和输出的相关信息，请参阅 [RoBERTa 的文档](roberta)。  

此模型由 [stefan-it](https://huggingface.co/stefan-it) 贡献。原始代码可在 [此处](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) 找到。

## 资源

以下是官方 Hugging Face 和社区（以🌎表示）资源列表，可帮助您开始使用 XLM-RoBERTa。如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将进行审查！该资源应该展示出新的东西，而不是重复现有的资源。
<PipelineTag pipeline="text-classification"/>
- 有关如何 [使用 Habana Gaudi 在 AWS 上微调 XLM RoBERTa 进行多类分类的博客文章](https://www.philschmid.de/habana-distributed-training)
- [`XLMRobertaForSequenceClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) 支持。
- [`TFXLMRobertaForSequenceClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb) 支持。
- [`FlaxXLMRobertaForSequenceClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb) 支持。
- 🤗 Hugging Face 任务指南的 [文本分类](https://huggingface.co/docs/transformers/tasks/sequence_classification) 章节。
- [文本分类任务指南](../tasks/sequence_classification)
<PipelineTag pipeline="token-classification"/>
- [`XLMRobertaForTokenClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) 支持。
- [`TFXLMRobertaForTokenClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb) 支持。
- [`FlaxXLMRobertaForTokenClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification) 支持。
- 🤗 Hugging Face 课程中的 [标记分类](https://huggingface.co/course/chapter7/2?fw=pt) 章节。
- [标记分类任务指南](../tasks/token_classification)
<PipelineTag pipeline="text-generation"/>
- [`XLMRobertaForCausalLM`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 支持。
- 🤗 Hugging Face 任务指南的 [因果语言建模](https://huggingface.co/docs/transformers/tasks/language_modeling) 章节。
- [因果语言建模任务指南](../tasks/language_modeling)
<PipelineTag pipeline="fill-mask"/>
- [`XLMRobertaForMaskedLM`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 中得到支持。
- [`TFXLMRobertaForMaskedLM`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb) 中得到支持。
- [`FlaxXLMRobertaForMaskedLM`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb) 中得到支持。- [遮蔽语言建模](https://huggingface.co/course/chapter7/3?fw=pt) 章节的🤗 Hugging Face 课程。
- [遮蔽语言建模](../tasks/masked_language_modeling)
<PipelineTag pipeline="question-answering"/>

- [`XLMRobertaForQuestionAnswering`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) 中得到支持。
- [`TFXLMRobertaForQuestionAnswering`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb) 中得到支持。
- [`FlaxXLMRobertaForQuestionAnswering`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering) 中得到支持。
- [问题回答](https://huggingface.co/course/chapter7/7?fw=pt) 章节的🤗 Hugging Face 课程。
- [问题回答任务指南](../tasks/question_answering)

**多项选择**

- [`XLMRobertaForMultipleChoice`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) 中得到支持。
- [`TFXLMRobertaForMultipleChoice`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb) 中得到支持。
- [多项选择任务指南](../tasks/multiple_choice)

🚀 部署

- 如何在 AWS Lambda 上 [部署无服务器 XLM RoBERTa](https://www.philschmid.de/multilingual-serverless-xlm-roberta-with-huggingface) 的博文。
## XLMRobertaConfig

[[autodoc]] XLMRobertaConfig

## XLMRobertaTokenizer

[[autodoc]] XLMRobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLMRobertaTokenizerFast

[[autodoc]] XLMRobertaTokenizerFast

## XLMRobertaModel

[[autodoc]] XLMRobertaModel
    - forward

## XLMRobertaForCausalLM

[[autodoc]] XLMRobertaForCausalLM
    - forward

## XLMRobertaForMaskedLM

[[autodoc]] XLMRobertaForMaskedLM
    - forward

## XLMRobertaForSequenceClassification

[[autodoc]] XLMRobertaForSequenceClassification
    - forward

## XLMRobertaForMultipleChoice

[[autodoc]] XLMRobertaForMultipleChoice
    - forward

## XLMRobertaForTokenClassification

[[autodoc]] XLMRobertaForTokenClassification
    - forward

## XLMRobertaForQuestionAnswering

[[autodoc]] XLMRobertaForQuestionAnswering
    - forward

## TFXLMRobertaModel

[[autodoc]] TFXLMRobertaModel
    - call

## TFXLMRobertaForCausalLM

[[autodoc]] TFXLMRobertaForCausalLM
    - call

## TFXLMRobertaForMaskedLM

[[autodoc]] TFXLMRobertaForMaskedLM
    - call

## TFXLMRobertaForSequenceClassification

[[autodoc]] TFXLMRobertaForSequenceClassification
    - call

## TFXLMRobertaForMultipleChoice

[[autodoc]] TFXLMRobertaForMultipleChoice
    - call

## TFXLMRobertaForTokenClassification

[[autodoc]] TFXLMRobertaForTokenClassification
    - call

## TFXLMRobertaForQuestionAnswering

[[autodoc]] TFXLMRobertaForQuestionAnswering
    - call

## FlaxXLMRobertaModel

[[autodoc]] FlaxXLMRobertaModel
    - __call__

## FlaxXLMRobertaForCausalLM

[[autodoc]] FlaxXLMRobertaForCausalLM
    - __call__

## FlaxXLMRobertaForMaskedLM

[[autodoc]] FlaxXLMRobertaForMaskedLM
    - __call__

## FlaxXLMRobertaForSequenceClassification

[[autodoc]] FlaxXLMRobertaForSequenceClassification
    - __call__

## FlaxXLMRobertaForMultipleChoice

[[autodoc]] FlaxXLMRobertaForMultipleChoice
    - __call__

## FlaxXLMRobertaForTokenClassification

[[autodoc]] FlaxXLMRobertaForTokenClassification
    - __call__

## FlaxXLMRobertaForQuestionAnswering

[[autodoc]] FlaxXLMRobertaForQuestionAnswering
    - __call__

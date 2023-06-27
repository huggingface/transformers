<!--版权所有 2023 年 The HuggingFace 和 Baidu 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件将按“按原样” BASIS，无论是明示还是暗示，都没有任何形式的保证或条件。请参阅许可证的特定语言的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->
# ErnieM

## 概述

ErnieM 模型是由 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang 提出的 [ERNIE-M：通过对齐跨语言语义与单语语料库来增强多语言表示](https://arxiv.org/abs/2012.15674)。

论文中的摘要如下：

*最近的研究表明，预训练的跨语言模型在下游跨语言任务中取得了令人印象深刻的性能。这一改进得益于学习大量的单语和平行语料库。尽管人们普遍认为平行语料对于提高模型性能至关重要，但现有方法通常受到平行语料大小的限制，特别是对于低资源语言。在本文中，我们提出了 ERNIE-M，一种新的训练方法，鼓励模型使用单语语料库对多种语言的表示进行对齐，以克服平行语料大小对模型性能的限制。我们的关键见解是将反向翻译集成到预训练过程中。我们在单语语料库上生成伪平行句对，以实现不同语言之间的语义对齐学习，从而增强跨语言模型的语义建模。实验结果表明，ERNIE-M 优于现有的跨语言模型，在各种跨语言下游任务中提供了新的最新结果。*

提示：

1. Ernie-M 是一种类似 BERT 的模型，因此它是一个堆叠的 Transformer 编码器。
2. 作者使用了两种新技术而不是像 BERT 一样使用 MaskedLM 进行预训练：`交叉注意力掩码语言建模` 和 `反向翻译掩码语言建模`。目前这两个 LMHead 目标在这里尚未实现。
3. 它是一个多语言语言模型。4. 预训练过程中没有使用下一个句子预测。

此模型由 [Susnato Dhar](https://huggingface.co/susnato) 贡献。原始代码可以在 [这里](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_m) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [多选任务指南](../tasks/multiple_choice)

## ErnieMConfig

[[autodoc]] ErnieMConfig


## ErnieMTokenizer

[[autodoc]] ErnieMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## ErnieMModel

[[autodoc]] ErnieMModel
    - forward

## ErnieMForSequenceClassification

[[autodoc]] ErnieMForSequenceClassification
    - forward


## ErnieMForMultipleChoice

[[autodoc]] ErnieMForMultipleChoice
    - forward


## ErnieMForTokenClassification

[[autodoc]] ErnieMForTokenClassification
    - forward


## ErnieMForQuestionAnswering

[[autodoc]] ErnieMForQuestionAnswering
    - forward

## ErnieMForInformationExtraction

[[autodoc]] ErnieMForInformationExtraction
    - forward

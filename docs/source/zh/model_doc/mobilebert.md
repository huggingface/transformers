<!--版权 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）获得许可；您除非符合许可证的规定，否则不得使用此文件。您可以在以下网址
http://www.apache.org/licenses/LICENSE-2.0
根据适用法律或书面同意，根据许可证分发的软件以“按原样” BASIS，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式的，但包含特定于我们 doc-builder 的语法（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。
-->
# MobileBERT

## 概述

MobileBERT 模型是由 Zhiqing Sun、Hongkun Yu、Xiaodan Song、Renjie Liu、Yiming Yang 和 Denny Zhou 在 [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) 中提出的。它是基于 BERT 模型的双向变压器，经过多种方法进行了压缩和加速。

以下是论文中的摘要内容: 

*然语言处理（NLP）最近通过使用具有数亿参数的大型预训练模型取得了巨大成功。然而，这些模型的体积较大，延迟较高，因此无法部署到资源有限的移动设备上。在本文中，我们提出了 MobileBERT 来压缩和加速流行的 BERT 模型。与原始的 BERT 一样，MobileBERT 是任务不可知的，也就是说，它可以通过简单的微调应用于各种下游 NLP 任务。基本上，MobileBERT 是 BERT_LARGE 的轻量版本，同时配备了瓶颈结构和经过精心设计的自注意力和前馈网络之间的平衡。
为了训练 MobileBERT，我们首先训练了一个特别设计的教师模型，即融入了倒置瓶颈的 BERT_LARGE 模型。然后，我们从这个教师模型向 MobileBERT 进行知识转移。实证研究表明，MobileBERT 的体积比 BERT_BASE 小 4.3 倍，速度比 BERT_BASE 快 5.5 倍，同时在众所周知的基准测试中取得了竞争力的结果。在 GLUE 的自然语言推理任务中，MobileBERT 获得了 77.7 的 GLUE 分数（比 BERT_BASE 低 0.6），在 Pixel 4 手机上的延迟为 62 毫秒。在 SQuAD v1.1/v2.0 的问答任务中，MobileBERT 的开发 F1 得分为 90.0/79.2（比 BERT_BASE 高 1.5/2.1）.*

此模型由 [vshampor](https://huggingface.co/vshampor) 贡献。原始代码可在 [此处](https://github.com/google-research/mobilebert) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## MobileBertConfig

[[autodoc]] MobileBertConfig

## MobileBertTokenizer

[[autodoc]] MobileBertTokenizer

## MobileBertTokenizerFast

[[autodoc]] MobileBertTokenizerFast

## MobileBert specific outputs

[[autodoc]] models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput

[[autodoc]] models.mobilebert.modeling_tf_mobilebert.TFMobileBertForPreTrainingOutput

## MobileBertModel

[[autodoc]] MobileBertModel
    - forward

## MobileBertForPreTraining

[[autodoc]] MobileBertForPreTraining
    - forward

## MobileBertForMaskedLM

[[autodoc]] MobileBertForMaskedLM
    - forward

## MobileBertForNextSentencePrediction

[[autodoc]] MobileBertForNextSentencePrediction
    - forward

## MobileBertForSequenceClassification

[[autodoc]] MobileBertForSequenceClassification
    - forward

## MobileBertForMultipleChoice

[[autodoc]] MobileBertForMultipleChoice
    - forward

## MobileBertForTokenClassification

[[autodoc]] MobileBertForTokenClassification
    - forward

## MobileBertForQuestionAnswering

[[autodoc]] MobileBertForQuestionAnswering
    - forward

## TFMobileBertModel

[[autodoc]] TFMobileBertModel
    - call

## TFMobileBertForPreTraining

[[autodoc]] TFMobileBertForPreTraining
    - call

## TFMobileBertForMaskedLM

[[autodoc]] TFMobileBertForMaskedLM
    - call

## TFMobileBertForNextSentencePrediction

[[autodoc]] TFMobileBertForNextSentencePrediction
    - call

## TFMobileBertForSequenceClassification

[[autodoc]] TFMobileBertForSequenceClassification
    - call

## TFMobileBertForMultipleChoice

[[autodoc]] TFMobileBertForMultipleChoice
    - call

## TFMobileBertForTokenClassification

[[autodoc]] TFMobileBertForTokenClassification
    - call

## TFMobileBertForQuestionAnswering

[[autodoc]] TFMobileBertForQuestionAnswering
    - call

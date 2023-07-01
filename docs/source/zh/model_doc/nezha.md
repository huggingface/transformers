<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据Apache许可证2.0版（“许可证”）获得许可，除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。请注意，此文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能无法在您的Markdown查看器中正确显示。
⚠️ 请注意，此文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能无法在您的Markdown查看器中正确显示。请注意，此文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能无法在您的Markdown查看器中正确显示。
-->
# Nezha

## 概述

Nezha模型是由魏俊秋等人在[NEZHA: 用于中文语言理解的神经上下文表示](https://arxiv.org/abs/1909.00204)一文中提出的。

该文摘要如下：

*由于预训练语言模型能够通过对大规模语料库进行预训练来捕捉文本中的深层上下文信息，因此在各种自然语言理解（NLU）任务中，预训练语言模型取得了巨大的成功。本技术报告中，我们介绍了我们在中文语料库上预训练的语言模型NEZHA（NEural contexualiZedrepresentation for CHinese lAnguage understanding）和其在中文NLU任务中的微调实践。当前版本的NEZHA基于BERT，并包含了一系列经过验证的改进，包括功能性相对位置编码作为一种有效的位置编码方案，全词掩蔽策略，混合精度训练和LAMB优化器等。实验结果表明，NEZHA在几个代表性的中文任务上进行微调后达到了最先进的性能，包括命名实体识别（人民日报NER），句子匹配（LCQMC），中文情感分类（ChnSenti）和自然语言推理（XNLI）.*

该模型由[sijunhe](https://huggingface.co/sijunhe)贡献。原始代码可在[此处](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch)找到。

## 文档资源
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [掩蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)
## NezhaConfig

[[autodoc]] NezhaConfig

## NezhaModel

[[autodoc]] NezhaModel
    - forward

## NezhaForPreTraining

[[autodoc]] NezhaForPreTraining
    - forward

## NezhaForMaskedLM

[[autodoc]] NezhaForMaskedLM
    - forward

## NezhaForNextSentencePrediction

[[autodoc]] NezhaForNextSentencePrediction
    - forward

## NezhaForSequenceClassification

[[autodoc]] NezhaForSequenceClassification
    - forward

## NezhaForMultipleChoice

[[autodoc]] NezhaForMultipleChoice
    - forward

## NezhaForTokenClassification

[[autodoc]] NezhaForTokenClassification
    - forward

## NezhaForQuestionAnswering

[[autodoc]] NezhaForQuestionAnswering
    - forward
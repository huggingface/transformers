<!--版权2022年HuggingFace团队。保留所有权利。
根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是根据“按原样”基础分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是Markdown格式，但包含了我们文档生成器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确渲染。
-->
# RoCBert

## 概述

RoCBert模型是由HuiSu、WeiweiShi、XiaoyuShen、XiaoZhou、TuoJi、JiaruiFang、JieZhou在[《RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining》](https://aclanthology.org/2022.acl-long.65.pdf)中提出的。这是一个预训练的中文语言模型，能够在各种形式的对抗攻击下保持稳健。

论文中的摘要如下：

*大规模预训练语言模型在自然语言处理任务中取得了SOTA结果。然而，它们对于象形文字语言（如中文）尤其容易受到对抗攻击。在这项工作中，我们提出了ROCBERT：一个预训练的中文Bert模型，能够抵抗各种形式的对抗攻击，如词汇扰动、同义词、错别字等。它通过对不同的合成对抗样本进行对比学习来最大化标签一致性在不同的模态信息，包括语义、音韵和视觉特征上进行预训练。我们展示了所有这些特征对模型的鲁棒性都很重要，因为攻击可以以这三种形式进行。在5个中文自然语言理解任务中，ROCBERT在三种黑盒对抗算法下表现优于强基线模型，同时不损失在干净测试集上的性能。在由人工制作的攻击下，它也在有毒内容检测任务中表现最好。*

此模型由[weiweishi](https://huggingface.co/weiweishi)贡献。


## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## RoCBertConfig

[[autodoc]] RoCBertConfig
    - all


## RoCBertTokenizer

[[autodoc]] RoCBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## RoCBertModel

[[autodoc]] RoCBertModel
    - forward


## RoCBertForPreTraining

[[autodoc]] RoCBertForPreTraining
    - forward


## RoCBertForCausalLM

[[autodoc]] RoCBertForCausalLM
    - forward


## RoCBertForMaskedLM

[[autodoc]] RoCBertForMaskedLM
    - forward


## RoCBertForSequenceClassification

[[autodoc]] transformers.RoCBertForSequenceClassification
    - forward

## RoCBertForMultipleChoice

[[autodoc]] transformers.RoCBertForMultipleChoice
    - forward


## RoCBertForTokenClassification

[[autodoc]] transformers.RoCBertForTokenClassification
    - forward


## RoCBertForQuestionAnswering

[[autodoc]] RoCBertForQuestionAnswering
    - forward
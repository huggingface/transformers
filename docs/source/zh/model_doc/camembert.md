<!--版权所有2020年The HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）的规定，您只能在符合以下条件的情况下使用此文件许可证。您可以在以下链接处获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律或书面同意，按“现状”分发的软件根据许可证分发基础，不提供任何明示或暗示的保证或条件。有关许可证的详细信息特定语言的权限和限制，请参阅许可证。
⚠️ 请注意，此文件是 Markdown 格式的，但包含特定于我们的 doc-builder（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# CamemBERT

## 概述

CamemBERT 模型是由 Louis Martin、Benjamin Muller、Pedro Javier Ortiz Su á rez、Yoann Dupont、Laurent Romary、É ric Villemonte de laClergerie、Djam é Seddah 和 Beno î t Sagot 在 [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) 一文中提出的。它基于 Facebook 于 2019 年发布的 RoBERTa 模型。它是在 138GB 的法语文本上训练的模型。

以下是论文的摘要内容：

*预训练语言模型在自然语言处理中已经无处不在。尽管它们成功了，但大多数可用的模型要么是在英文数据上训练的，要么是在多种语言的数据上进行拼接的。这使得：

在除了英语以外的所有语言中实际使用这些模型非常有限。为了解决法语这个问题，我们发布了 CamemBERT，它是 Bi-directional Encoders for Transformers (BERT)的法语版本。我们通过多个下游任务，包括词性标注、依存句法分析、命名实体识别和自然语言推理，对 CamemBERT 的性能进行了衡量。CamemBERT 在大多数考虑的任务中改善了现有技术水平。我们发布了 CamemBERT 的预训练模型，希望能够促进法语自然语言处理的研究和下游应用。*
提示：


- 这个实现与 RoBERTa 相同。有关用法示例以及输入和输出的相关信息，请参考 [RoBERTa 的文档](roberta)。  

此模型由 [camembert](https://huggingface.co/camembert) 贡献。原始代码可以在 [这里](https://camembert-model.fr/) 找到。

## 文档资源


- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## CamembertConfig

[[autodoc]] CamembertConfig

## CamembertTokenizer

[[autodoc]] CamembertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CamembertTokenizerFast

[[autodoc]] CamembertTokenizerFast

## CamembertModel

[[autodoc]] CamembertModel

## CamembertForCausalLM

[[autodoc]] CamembertForCausalLM

## CamembertForMaskedLM

[[autodoc]] CamembertForMaskedLM

## CamembertForSequenceClassification

[[autodoc]] CamembertForSequenceClassification

## CamembertForMultipleChoice

[[autodoc]] CamembertForMultipleChoice

## CamembertForTokenClassification

[[autodoc]] CamembertForTokenClassification

## CamembertForQuestionAnswering

[[autodoc]] CamembertForQuestionAnswering

## TFCamembertModel

[[autodoc]] TFCamembertModel

## TFCamembertForCasualLM

[[autodoc]] TFCamembertForCausalLM

## TFCamembertForMaskedLM

[[autodoc]] TFCamembertForMaskedLM

## TFCamembertForSequenceClassification

[[autodoc]] TFCamembertForSequenceClassification

## TFCamembertForMultipleChoice

[[autodoc]] TFCamembertForMultipleChoice

## TFCamembertForTokenClassification

[[autodoc]] TFCamembertForTokenClassification

## TFCamembertForQuestionAnswering

[[autodoc]] TFCamembertForQuestionAnswering


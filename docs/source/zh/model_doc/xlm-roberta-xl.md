<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（"许可证"）获得许可; 除非符合许可证的要求，否则您无权使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，不附带任何形式的担保或条件。请参阅许可证以了解具体语言下的权限和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
注意：此文件为 Markdown 格式，但包含了我们文档生成器的特定语法（类似于 MDX），可能在您的 Markdown 查看器中无法正确呈现。
⚠️ 请注意，此文件是 Markdown 格式，但包含了我们文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。渲染示例如下：
-->
# XLM-RoBERTa-XL

## 概述

XLM-RoBERTa-XL 模型是由 Naman Goyal、Jingfei Du、Myle Ott、Giri Anantharaman、Alexis Conneau 在论文 [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/abs/2105.00572) 中提出的。

论文摘要如下：

*最近的研究表明，跨语言语言模型预训练对于跨语言理解非常有效。在本研究中，我们提出了两个更大的多语言遮掩语言模型，参数为 35 亿和 107 亿。我们的两个新模型命名为 XLM-R XL 和 XLM-R XXL，在 XNLI 上的平均准确率比 XLM-R 高出 1.8%和 2.4%。与 RoBERTa-Large 模型相比，我们的模型在 GLUE 基准测试的几个英文任务上的平均准确率提高了 0.3%，同时处理了 99 种更多的语言。这表明具有更大容量的预训练模型可以在高资源语言上获得强大的性能，同时大大提高低资源语言的性能。我们将我们的代码和模型公开提供。*

提示：

- XLM-RoBERTa-XL 是一个在 100 种不同语言上训练的多语言模型。与一些 XLM 多语言模型不同，它不需要 `lang` 张量来确定使用的语言，并且应该能够从输入的 id 中确定正确的语言。  语言。这个模型是由 [Soonhwan-Kwon](https://github.com/Soonhwan-Kwon) 和 [stefan-it](https://huggingface.co/stefan-it) 贡献的。原始代码可以在 [这里](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) 找到。



## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [遮掩语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## XLMRobertaXLConfig

[[autodoc]] XLMRobertaXLConfig

## XLMRobertaXLModel

[[autodoc]] XLMRobertaXLModel
    - forward

## XLMRobertaXLForCausalLM

[[autodoc]] XLMRobertaXLForCausalLM
    - forward

## XLMRobertaXLForMaskedLM

[[autodoc]] XLMRobertaXLForMaskedLM
    - forward

## XLMRobertaXLForSequenceClassification

[[autodoc]] XLMRobertaXLForSequenceClassification
    - forward

## XLMRobertaXLForMultipleChoice

[[autodoc]] XLMRobertaXLForMultipleChoice
    - forward

## XLMRobertaXLForTokenClassification

[[autodoc]] XLMRobertaXLForTokenClassification
    - forward

## XLMRobertaXLForQuestionAnswering

[[autodoc]] XLMRobertaXLForQuestionAnswering
    - forward
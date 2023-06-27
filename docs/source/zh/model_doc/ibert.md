<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”），您不得使用此文件，除非符合许可证。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的基础，不附带任何形式的保证或条件。请参阅许可证特定语言的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# I-BERT

## 概览

I-BERT 模型是由 Sehoon Kim、Amir Gholami、Zhewei Yao、Michael W. Mahoney 和 Kurt Keutzer 在 [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) 中提出的。

它是 RoBERTa 的量化版本，推理速度最多快四倍。以下是论文的摘要: 

*基于 Transformer 的模型，例如 BERT 和 RoBERTa，在许多自然语言处理任务中取得了最先进的结果。然而，它们的内存占用，推理延迟和功耗都不适合在边缘进行高效推理，甚至在数据中心也是如此。虽然量化可以成为解决方案，但先前的基于 Transformer 的模型量化工作在推理过程中使用浮点运算，这不能有效地利用整数逻辑单元，例如最近的图灵张量核心或传统的整数逻辑单元处理器。在这项工作中，我们提出了 I-BERT，一种新颖的 Transformer 模型量化方案，可以将整个推理过程用整数运算来量化。基于轻量级整数逼近方法进行非线性操作例如 GELU，Softmax 和 Layer Normalization，I-BERT 执行一种纯整数的 BERT 推理，不进行任何浮点计算。我们使用 RoBERTa-Base/Large 在 GLUE 下游任务上评估了我们的方法。结果表明，对于这两种情况，与完全精度的基准相比，I-BERT 实现了相似（稍微更高）的准确性。此外，我们对 T4 GPU 系统上的 INT8 推理的初步实现显示出 2.4-4.0 倍的加速。该框架已经在 PyTorch 中开发并开源。* 

此模型由 [kssteven](https://huggingface.co/kssteven) 贡献。原始代码可以在 [这里](https://github.com/kssteven418/I-BERT) 找到。



## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/masked_language_modeling)

## IBertConfig

[[autodoc]] IBertConfig

## IBertModel

[[autodoc]] IBertModel
    - forward

## IBertForMaskedLM

[[autodoc]] IBertForMaskedLM
    - forward

## IBertForSequenceClassification

[[autodoc]] IBertForSequenceClassification
    - forward

## IBertForMultipleChoice

[[autodoc]] IBertForMultipleChoice
    - forward

## IBertForTokenClassification

[[autodoc]] IBertForTokenClassification
    - forward

## IBertForQuestionAnswering

[[autodoc]] IBertForQuestionAnswering
    - forward

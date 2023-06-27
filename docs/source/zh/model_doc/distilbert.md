<!--版权所有2020年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）许可；您除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件在许可证下分发。不提供任何形式的明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。⚠️请注意，该文件是 Markdown 格式，但包含我们的文档构建工具的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。特性语言的授权和限制。


-->

# DistilBERT

<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=distilbert"> <img alt="模型" src="https://img.shields.io/badge/All_model_pages-distilbert-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/distilbert-base-uncased"> <img alt="空间" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> <a href="https://huggingface.co/papers/1910.01108"> <img alt="论文页面" src="https://img.shields.io/badge/Paper%20page-1910.01108-green"> </a> </div>

## 概述

DistilBERT 模型是在 [Smaller, faster, cheaper, lighter: Introducing DistilBERT, adistilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5) 博文和 [DistilBERT, adistilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/papers/1910.01108) 论文中提出的。

DistilBERT 是通过精简 BERT base 训练得到的一个小型、快速、廉价和轻量级的 Transformer 模型。与 *bert-base-uncased* 相比，它的参数数量减少了 40%，运行速度提高了 60%，同时保留了在 GLUE 语言理解基准测试上超过 95%的 BERT 性能。以下是论文的摘要内容:


*随着从大规模预训练模型的迁移学习在自然语言处理（NLP）中变得越来越普遍，将这些大模型应用于边缘设备和/或受限计算训练或推理预算的情况仍然具有挑战性。在这项工作中，我们提出了一种方法，预训练一个更小的通用语言表示模型，称为 DistilBERT，然后在多个任务上进行微调，达到与更大的模型相当的性能。虽然大多数先前的工作研究了使用蒸馏建立特定任务模型的方法，但我们利用预训练阶段的知识蒸馏，并表明可以将 BERT 模型的大小减小 40%，同时保留 97%的语言理解能力，运行速度提高 60%。为了利用在预训练期间较大模型学到的归纳偏差，我们引入了一个三元损失，结合了语言建模、蒸馏和余弦距离损失。我们更小、更快、更轻的模型在预训练阶段更便宜，我们通过概念验证实验和比较在设备上的研究证明了它的能力。* 

技巧：

- DistilBERT 没有 `token_type_ids`，您不需要指示哪个标记属于哪个段落。只需使用分隔标记 `tokenizer.sep_token`（或 `[SEP]`）分隔段落即可。

- DistilBERT 没有选择输入位置（`position_ids` 输入）的选项。如果有必要，可以添加此选项，只需告诉我们您需要此选项。

- 与 BERT 相同但更小。通过对预训练的 BERT 模型进行蒸馏训练，意味着它已经被训练成预测与较大模型相同的概率。实际目标是：
    * 寻找与教师模型相同的概率    
    * 正确预测掩码标记（但没有下一个句子的目标）    
    * 学生模型和教师模型的隐藏状态之间的余弦相似性

此模型由 [victorsanh](https://huggingface.co/victorsanh) 贡献。此模型的 jax 版本由 [kamalkraj](https://huggingface.co/kamalkraj) 贡献。原始代码可在 [此处](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源列表，以帮助您开始使用 DistilBERT。如果您有兴趣提交资源以包含在此处，请随时打开一个 Pull Request，我们会进行审核！该资源应该展示出新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 一篇关于 [使用 Python 进行情感分析入门](https://huggingface.co/blog/sentiment-analysis-python) 的博文，使用 DistilBERT。
- 一篇关于如何使用 Blurr 对 DistilBERT 进行序列分类训练的博文。- 一篇关于如何使用 Ray 调整 DistilBERT 超参数的博文。
- 一篇关于如何使用 Hugging Face 和 Amazon SageMaker 训练 DistilBERT 的博文。
- 一份关于如何为多标签分类微调 DistilBERT 的笔记本。
🌎- 一份关于如何使用 PyTorch 为多类分类微调 DistilBERT 的笔记本。
🌎- 一份关于如何在 TensorFlow 中为文本分类微调 DistilBERT 的笔记本。
🌎- 此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) 支持 `DistilBertForSequenceClassification`。
- 此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb) 支持 `TFDistilBertForSequenceClassification`。
- 此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb) 支持 `FlaxDistilBertForSequenceClassification`。
- [文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>
- 此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) 支持 `DistilBertForTokenClassification`。
- [`TFDistilBertForTokenClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb) 支持。
- [`FlaxDistilBertForTokenClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification) 支持。
- [令牌分类](https://huggingface.co/course/chapter7/2?fw=pt) 章节的🤗 Hugging Face 课程。
- [令牌分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`DistilBertForMaskedLM`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 支持。
- [`TFDistilBertForMaskedLM`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb) 支持。
- [`FlaxDistilBertForMaskedLM`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) 支持。
- [掩码语言建模](https://huggingface.co/course/chapter7/3?fw=pt) 章节的🤗 Hugging Face 课程。
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
<PipelineTag pipeline="question-answering"/>
- [`DistilBertForQuestionAnswering`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) 支持。
- [`TFDistilBertForQuestionAnswering`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb) 支持。
- [`FlaxDistilBertForQuestionAnswering`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering) 支持。
- [问答](https://huggingface.co/course/chapter7/7?fw=pt) 章节的🤗 Hugging Face 课程。
- [问答任务指南](../tasks/question_answering)

**多项选择**

- [`DistilBertForMultipleChoice`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) 支持。
- [`TFDistilBertForMultipleChoice`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb) 支持。
- [多项选择任务指南](../tasks/multiple_choice)
⚗️ 优化
- 有关如何使用🤗 Optimum 和 Intel 对 DistilBERT 进行 [量化](https://huggingface.co/blog/intel) 的博文。
- 关于如何使用🤗 Optimum 对 Transformers 进行 [GPU 优化](https://www.philschmid.de/optimizing-transformers-with-optimum-gpu) 的博文。
- 有关使用 Hugging Face Optimum 对 Transformers 进行 [优化](https://www.philschmid.de/optimizing-transformers-with-optimum) 的博文。
⚡️ 推理
- 有关如何使用 Hugging Face Transformers 和 AWS Inferentia 加速 BERT [推理](https://huggingface.co/blog/bert-inferentia-sagemaker) 的博文，其中使用了 DistilBERT。
- 关于使用 Hugging Face Transformers、DistilBERT 和 Amazon SageMaker 进行无服务器 [推理](https://www.philschmid.de/sagemaker-serverless-huggingface-distilbert) 的博文。

🚀 部署
- 有关如何在 Google Cloud 上 [部署](https://huggingface.co/blog/how-to-deploy-a-pipeline-to-google-clouds) DistilBERT 的博文。
- 有关如何使用 Amazon SageMaker [部署](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker) DistilBERT 的博文。
- 有关如何使用 Hugging Face Transformers、Amazon SageMaker 和 Terraform 模块 [部署](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker) BERT 的博文。

## DistilBertConfig

[[autodoc]] DistilBertConfig

## DistilBertTokenizer

[[autodoc]] DistilBertTokenizer

## DistilBertTokenizerFast

[[autodoc]] DistilBertTokenizerFast

## DistilBertModel

[[autodoc]] DistilBertModel
    - forward

## DistilBertForMaskedLM

[[autodoc]] DistilBertForMaskedLM
    - forward

## DistilBertForSequenceClassification

[[autodoc]] DistilBertForSequenceClassification
    - forward

## DistilBertForMultipleChoice

[[autodoc]] DistilBertForMultipleChoice
    - forward

## DistilBertForTokenClassification

[[autodoc]] DistilBertForTokenClassification
    - forward

## DistilBertForQuestionAnswering

[[autodoc]] DistilBertForQuestionAnswering
    - forward

## TFDistilBertModel

[[autodoc]] TFDistilBertModel
    - call

## TFDistilBertForMaskedLM

[[autodoc]] TFDistilBertForMaskedLM
    - call

## TFDistilBertForSequenceClassification

[[autodoc]] TFDistilBertForSequenceClassification
    - call

## TFDistilBertForMultipleChoice

[[autodoc]] TFDistilBertForMultipleChoice
    - call

## TFDistilBertForTokenClassification

[[autodoc]] TFDistilBertForTokenClassification
    - call

## TFDistilBertForQuestionAnswering

[[autodoc]] TFDistilBertForQuestionAnswering
    - call

## FlaxDistilBertModel

[[autodoc]] FlaxDistilBertModel
    - __call__

## FlaxDistilBertForMaskedLM

[[autodoc]] FlaxDistilBertForMaskedLM
    - __call__

## FlaxDistilBertForSequenceClassification

[[autodoc]] FlaxDistilBertForSequenceClassification
    - __call__

## FlaxDistilBertForMultipleChoice

[[autodoc]] FlaxDistilBertForMultipleChoice
    - __call__

## FlaxDistilBertForTokenClassification

[[autodoc]] FlaxDistilBertForTokenClassification
    - __call__

## FlaxDistilBertForQuestionAnswering

[[autodoc]] FlaxDistilBertForQuestionAnswering
    - __call__

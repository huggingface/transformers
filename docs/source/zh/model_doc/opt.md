<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非遵守许可证的规定，否则不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件根据许可证进行分发，无论是明示还是暗示的，都没有任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件采用 Markdown 格式，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确显示。
-->
# OPT

## 概述

OPT 模型是 Meta AI 在 [Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068) 中提出的。OPT 是一系列开源的大规模因果语言模型，其性能与 GPT3 相似。

以下是论文中的摘要内容：

*大型语言模型通常经过数十万个计算日的训练，在零样本和少样本学习方面展现出卓越能力。考虑到它们的计算成本，如果没有大量的资本，很难复制这些模型。对于可通过 API 获得的模型，无法访问完整的模型权重，这使得它们难以研究。我们提出了 Open Pre-trained Transformers（OPT），这是一套仅由解码器组成的预训练 transformers，参数范围从 125M 到 175B，我们希望与感兴趣的研究人员全面而负责任地共享。我们展示了 OPT-175B 与 GPT-3 相媲美，同时只需 1/7 的碳足迹进行开发。我们还发布了详细记录我们面临的基础设施挑战的日志，以及用于对所有发布模型进行实验的代码。*

提示：- OPT 具有与 [`BartDecoder`] 相同的架构。- 与 GPT2 相反，OPT 在每个提示的开头添加了 EOS 标记 `</s>`。

此模型由 [Arthur Zucker](https://huggingface.co/ArthurZ)，[Younes Belkada](https://huggingface.co/ybelkada) 和 [Patrick Von Platen](https://huggingface.co/patrickvonplaten) 贡献。原始代码可以在 [这里](https://github.com/facebookresearch/metaseq) 找到。

## 资源
以下是官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 OPT。如果您有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审核。资源理想情况下应该展示一些新内容，而不是重复现有资源。
<PipelineTag pipeline="text-generation" />
- 有关 [使用 PEFT、bitsandbytes 和 Transformers 对 OPT 进行微调的笔记本](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing)。🌎- 有关 [使用 OPT 的解码策略的博客文章](https://huggingface.co/blog/introducing-csearch#62-example-two---opt)。
- 🤗 Hugging Face 课程中的 [因果语言建模](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) 章节。- [`OPTForCausalLM`] 支持此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)。- [`TFOPTForCausalLM`] 支持此 [TensorFlow 因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)。- [`FlaxOPTForCausalLM`] 支持此 [Flax 因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling)。
<PipelineTag pipeline="text-classification" />
- [文本分类任务指南](sequence_classification.md)- [`OPTForSequenceClassification`] 支持此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)。
<PipelineTag pipeline="question-answering" />
- [`OPTForQuestionAnswering`] 支持此 [问答示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)。
- 🤗 Hugging Face 课程的 [问答章节](https://huggingface.co/course/chapter7/7?fw=pt)。  

⚡️推理

- 有关 [如何通过 PyTorch 运行非常大的模型的博客文章](https://huggingface.co/blog/accelerate-large-models)，其中包括 OPT。

## OPTConfig

[[autodoc]] OPTConfig

## OPTModel

[[autodoc]] OPTModel
    - forward

## OPTForCausalLM

[[autodoc]] OPTForCausalLM
    - forward

## TFOPTModel

[[autodoc]] TFOPTModel
    - call

## TFOPTForCausalLM

[[autodoc]] TFOPTForCausalLM
    - call

## OPTForSequenceClassification

[[autodoc]] OPTForSequenceClassification
    - forward

## OPTForQuestionAnswering

[[autodoc]] OPTForQuestionAnswering
    - forward

## FlaxOPTModel

[[autodoc]] FlaxOPTModel
    - __call__


## FlaxOPTForCausalLM

[[autodoc]] FlaxOPTForCausalLM
    - __call__

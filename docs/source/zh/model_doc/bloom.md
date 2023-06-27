<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；您不得使用此文件，除非符合许可证。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不提供任何明示或暗示的保证或条件。有关许可证的特定语言下的权限和限制，详细了解请参阅许可证。
⚠️ 请注意，此文件采用 Markdown 格式，但包含特定语法，用于我们的文档构建器（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。
-->
# BLOOM

## 概览

BLOOM 模型是通过 [BigScience Workshop](https://bigscience.huggingface.co/) 提出的，通过各种版本进行了改进。BigScience 受到其他开放科学倡议的启发，研究人员汇集时间和资源，共同实现更高的影响力。BLOOM 的架构与 GPT3（用于下一个标记预测的自回归模型）基本相似，但是它经过了对 46 种不同语言和 13 种编程语言的训练。该数据集还训练了该模型的几个较小版本。BLOOM 有以下版本可用：


- [bloom-560m](https://huggingface.co/bigscience/bloom-560m)

- [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
- [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bloom](https://huggingface.co/bigscience/bloom)（参数量为 176B）

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）提供的资源列表，可帮助您入门 BLOOM。如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将进行审核！资源应该展示出新的东西，而不是重复现有的资源。
<PipelineTag pipeline="text-generation"/>

- [`BloomForCausalLM`] 由此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 支持。

另请参阅：
- [因果语言建模任务指南](../tasks/language_modeling)
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)

⚡️ 推理

- 关于 [优化故事：BLOOM 推理](https://huggingface.co/blog/bloom-inference-optimization) 的博客。
- 关于 [使用 DeepSpeed 和 Accelerate 进行极速 BLOOM 推理](https://huggingface.co/blog/bloom-inference-pytorch-scripts) 的博客。

⚙️ 训练

- 关于 [BLOOM 训练背后的技术](https://huggingface.co/blog/bloom-megatron-deepspeed) 的博客。

## BloomConfig

[[autodoc]] BloomConfig
    - all

## BloomModel

[[autodoc]] BloomModel
    - forward

## BloomTokenizerFast

[[autodoc]] BloomTokenizerFast
    - all

## BloomForCausalLM

[[autodoc]] BloomForCausalLM
    - forward

## BloomForSequenceClassification

[[autodoc]] BloomForSequenceClassification
    - forward

## BloomForTokenClassification

[[autodoc]] BloomForTokenClassification
    - forward

## BloomForQuestionAnswering

[[autodoc]] BloomForQuestionAnswering
    - forward

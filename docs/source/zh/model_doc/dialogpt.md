<!--版权所有 2020 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非遵守许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。注意：此文件为 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法正确渲染在您的 Markdown 查看器中。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# DialoGPT
## 概述
DialoGPT 是由 Yizhe Zhang、Siqi Sun、Michel Galley、Yen-Chun Chen、Chris Brockett、Xiang Gao、Jianfeng Gao、Jingjing Liu 和 Bill Dolan 提出的 [GPT2 模型的大规模生成预训练对话](https://arxiv.org/abs/1911.00536)。它是在从 Reddit 中提取的 147M 个类似对话的交流中训练的。来自论文的摘要如下：Reddit.

*我们提出了一种大型、可调节的神经对话响应生成模型 DialoGPT（对话生成预训练转换器）。DialoGPT 在从 2005 年到 2017 年的 Reddit 评论链中提取的 147M 个类似对话的交流上进行训练，扩展了 Hugging Face PyTorch 转换器，以在单轮对话设置中实现接近人类的性能，无论是从自动评估还是从人工评估的角度来看。我们展示了利用 DialoGPT 的对话系统生成比强基线系统更相关、更具内容和上下文一致性的响应。已公开发布预训练模型和训练流程，以促进神经响应生成的研究以及更智能的开放领域对话系统的开发。*


提示：

- DialoGPT 是一个具有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧进行填充输入。
- DialoGPT 是使用因果语言建模（CLM）目标在会话型数据上训练的，因此在开放领域对话系统中生成响应非常强大。
- DialoGPT 使用户能够仅使用 10 行代码创建聊天机器人，如 [DialoGPT 模型卡片](https://huggingface.co/microsoft/DialoGPT-medium) 中所示。  

训练：

为了训练或微调 DialoGPT，可以使用因果语言建模训练。引用官方论文：*我们遵循 OpenAI GPT-2 的方法，将多轮对话会话建模为一段长文本，并将生成任务作为语言建模。我们首先将对话会话中的所有对话转换为一段长文本 x_1,...，x_N（N 为序列长度），以结束文本符号结尾。* 更多信息请参考原始论文。


DialoGPT 的架构基于 GPT2 模型，因此可以参考 [GPT2 的文档页面](gpt2)。

原始代码可以在 [此处](https://github.com/microsoft/DialoGPT) 找到。
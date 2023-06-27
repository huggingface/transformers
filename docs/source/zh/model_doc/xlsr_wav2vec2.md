<!--版权所有2021年HuggingFace团队保留所有权利。-->
根据 Apache 许可证 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
获取许可证的副本。除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，没有任何明示或暗示的担保或条件。请参阅许可证以了解
特定语言下的权限和限制。请注意，此文件采用 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。-->

# XLSR-Wav2Vec2# XLSR-Wav2Vec2

## 概述
## Overview

XLSR-Wav2Vec2 模型是由 Alexis Conneau、Alexei Baevski、Ronan Collobert、Abdelrahman Mohamed 和 MichaelAuli 在 [无监督的跨语言表征学习语音识别](https://arxiv.org/abs/2006.13979) 中提出的。Auli.

论文摘要如下：

*本文介绍了 XLSR 模型，它通过在多种语言的原始语音波形上预训练一个模型来学习跨语言的语音表征。我们基于 wav2vec 2.0 进行构建，该模型通过解决对遮蔽的潜在语音表征进行对比任务来训练，并且共同学习了跨语言的潜在语音表征的量化。结果模型在标记数据上进行微调，实验证明跨语言预训练明显优于单语预训练。在 CommonVoice 基准测试中，相对于已知的最佳结果，XLSR 的相对音素错误率减少了 72%。在 BABEL 上，与可比系统相比，我们的方法将词错误率提高了 16%。我们的方法实现了竞争力强的单一多语言语音识别模型。分析表明，潜在离散语音表征在语言之间是共享的，相关的语言之间的共享程度更高。我们希望通过发布在 53 种语言中预训练的大型模型 XLSR-53 来促进低资源语音理解的研究。* 

- XLSR-Wav2Vec2 是一个接受与语音信号的原始波形对应的浮点数组的语音模型。- XLSR-Wav2Vec2 模型使用连接主义时序分类（CTC）进行训练，因此模型输出必须使用 [`Wav2Vec2CTCTokenizer`] 进行解码。

XLSR-Wav2Vec2 的架构基于 Wav2Vec2 模型，因此可以参考 [Wav2Vec2 的文档页面](wav2vec2)。

可以在 [此处](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec) 找到原始代码。


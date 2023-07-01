<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下网址获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何明示或暗示的担保或条件。请参阅许可证，了解特定语言下的权限和限制。

⚠️请注意，此文件为 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->

# M-CTC-T

## 概述

M-CTC-T 模型是由 Loren Lugosch、Tatiana Likhomanenko、Gabriel Synnaeve 和 Ronan Collobert 在 [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161) 中提出的。

该模型是一个具有 8065 个字符标签的 1B 参数的 Transformer 编码器，具有一个 CTC 头部和一个包含 60 个语言 ID 标签的语言识别头部。它是在 Common Voice（版本 6.1，2020 年 12 月发布）和 VoxPopuli 上进行训练的。在训练 Common Voice 和 VoxPopuli 之后，该模型仅在 Common Voice 上进行训练。标签是未归一化的字符级转录（不删除标点符号和大写字母）。模型以来自 16Khz 音频信号的 Mel 滤波器组特征作为输入。

论文摘要如下:

*通过伪标签半监督学习已成为最先进的单语言语音识别系统的重要组成部分。在本工作中，我们将伪标签扩展到了包含 60 种语言的大规模多语言语音识别。我们提出了一种简单的伪标签配方，即使在资源稀缺的语言中也能很好地工作：训练一个监督的多语言模型，用目标语言进行半监督学习进行微调，为该语言生成伪标签，并使用所有语言的伪标签训练最终模型，可以从头开始训练或者通过微调进行训练。对标记的 Common Voice 和未标记的 VoxPopuli 数据集的实验表明，我们的方法可以获得比较好的性能的模型，并且在 LibriSpeech 上也具有良好的迁移性能。


该模型由 [cwkeam](https://huggingface.co/cwkeam) 贡献。原始代码可在 [此处](https://github.com/flashlight/wav2letter/tree/main/recipes/mling_pl) 找到。

## 文档资源

- [自动语音识别任务指南](../tasks/asr)

提示:

- 此模型的 PyTorch 版本仅在 torch 1.9 及更高版本中可用。

## MCTCTConfig

[[autodoc]] MCTCTConfig

## MCTCTFeatureExtractor

[[autodoc]] MCTCTFeatureExtractor
    - __call__

## MCTCTProcessor

[[autodoc]] MCTCTProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode


## MCTCTModel

[[autodoc]] MCTCTModel
    - forward

## MCTCTForCTC

[[autodoc]] MCTCTForCTC
    - forward

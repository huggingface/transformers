<!--版权2022年The HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）进行许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何形式的明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->

# TimeSformer## 概述

TimeSformer 模型是由 Facebook Research 在 [TimeSformer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) 中提出的。

这项工作是行动识别领域的里程碑，是第一个基于 Transformer 的视频模型。它激发了许多基于 Transformer 的视频理解和分类论文。

该论文的摘要如下所示：

*我们提出了一种无卷积的视频分类方法，完全基于时空自注意力。我们的方法名为“TimeSformer”，通过在帧级补丁序列上直接进行时空特征学习来适应标准的 Transformer 架构。我们的实验研究比较了不同的自注意机制，并表明在考虑的设计选择中，“分割注意力”即将时间注意力和空间注意力分别应用于每个块中，可以获得最佳的视频分类准确性。尽管设计完全不同，TimeSformer 在多个行动识别基准上实现了最先进的结果，包括在 Kinetics-400 和 Kinetics-600 上报告的最佳准确性。最后，与 3D 卷积网络相比，我们的模型训练速度更快，测试效率大大提高（准确性稍有下降），还可以应用于更长的视频剪辑（超过一分钟）。代码和模型可在以下位置获得：[this https URL](https://github.com/facebookresearch/TimeSformer)。*

提示：

有许多预训练的变体。根据模型训练的数据集选择适合的预训练模型。此外，每个剪辑的输入帧数会根据模型大小而变化，因此在选择预训练模型时应考虑此参数。

该模型由 [fcakyon](https://huggingface.co/fcakyon) 贡献。
可以在 [此处](https://github.com/facebookresearch/TimeSformer) 找到原始代码。

## 文档资源

- [视频分类任务指南](../tasks/video_classification) The original code can be found [here](https://github.com/facebookresearch/TimeSformer).

## TimesformerConfig

[[autodoc]] TimesformerConfig

## TimesformerModel

[[autodoc]] TimesformerModel
    - forward

## TimesformerForVideoClassification

[[autodoc]] TimesformerForVideoClassification
    - forward
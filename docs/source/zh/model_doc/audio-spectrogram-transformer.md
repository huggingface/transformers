<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件。请参阅许可证以了解许可证下的特定语言规定和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
注意：此文件为 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。渲染。
-->

# 音频频谱变换器

## 概述

音频频谱变换器模型最早由 Yuan Gong、Yu-An Chung 和 James Glass 在 [AST：音频频谱变换器](https://arxiv.org/abs/2104.01778) 中提出。音频频谱变换器通过将音频转化为图像（频谱图）并应用 [视觉变换器](vit) 到音频中，从而获得了最先进的结果。音频分类。
论文摘要如下：

*在过去的十年中，卷积神经网络（CNN）已被广泛采用作为端到端音频分类模型的主要构建块，旨在学习从音频频谱到对应标签的直接映射。为了更好地捕捉长程全局上下文，最近的趋势是在 CNN 之上添加自注意机制，形成 CNN-attention 混合模型。然而，目前尚不清楚是否有必要依赖 CNN，并且纯粹基于注意力的神经网络是否足以获得良好的音频分类性能。在本文中，我们通过引入音频频谱变换器（AST），第一个无卷积，纯注意力模型，回答了这个问题。我们在各种音频分类基准上评估了 AST，在 AudioSet 上实现了 0.485 mAP 的最新成果，在 ESC-50 上实现了 95.6 ％的准确率，在语音命令 V2 上实现了 98.1 ％的准确率。*

小贴士：

- 当在自己的数据集上对音频频谱变换器（AST）进行微调时，建议注意输入归一化（确保输入的均值为 0，标准差为 0.5）。 [`ASTFeatureExtractor`] 会处理这个。请注意，默认情况下，它使用 AudioSet 的均值和标准差。您可以查看 [`ast/src/get_norm_stats.py`](https://github.com/YuanGongND/ast/blob/master/src/get_norm_stats.py) 以了解作者如何计算下游数据集的统计信息。
- 请注意，AST 需要较低的学习率（作者在 [PSLA 论文](https://arxiv.org/abs/2102.01243) 中提出的 CNN 模型的学习率小了 10 倍），并且收敛迅速，请为您的任务寻找合适的学习率和学习率调度器。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/audio_spectogram_transformer_architecture.png"alt="drawing" width="600"/>

<small> 音频频谱变换器架构。来源于 <a href="https://arxiv.org/abs/2104.01778"> 原始论文 </a>。</small>

此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在 [这里](https://github.com/YuanGongND/ast) 找到。

## 资源
以下是官方 Hugging Face 和社区（由🌎表示）资源列表，以帮助您开始使用音频频谱变换器。

<PipelineTag pipeline="audio-classification"/>

- 可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/AST) 找到使用 AST 进行音频分类的推理示例笔记本。- [`ASTForAudioClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb) 支持。- 另请参阅：[音频分类](../tasks/audio_classification)。

如果您有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审查！该资源应该展示出与现有资源不同的新内容，而不是重复现有资源。

## ASTConfig

[[autodoc]] ASTConfig

## ASTFeatureExtractor

[[autodoc]] ASTFeatureExtractor
    - __call__

## ASTModel

[[autodoc]] ASTModel
    - forward

## ASTForAudioClassification

[[autodoc]] ASTForAudioClassification
    - forward
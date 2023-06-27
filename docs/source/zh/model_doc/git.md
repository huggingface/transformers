<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不能使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
根据适用法律或书面协议，根据许可证分发的软件是基于“按原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 文件，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法在 Markdown 查看器中正确呈现。
-->
# GIT

## 概述

GIT 模型是由 Jianfeng Wang、Zhengyuan Yang、Xiaowei Hu、Linjie Li、Kevin Lin、Zhe Gan、Zicheng Liu、Ce Liu 和 Lijuan Wang 在 [《GIT: 用于视觉和语言的生成式图像到文本转换器》](https://arxiv.org/abs/2205.14100) 中提出的。GIT 是一个仅具有解码器的 Transformer 模型，除了文本之外，还利用了 [CLIP](clip) 的视觉编码器对模型进行条件化。该模型在图像字幕和视觉问答基准上取得了最先进的结果。论文摘要如下：

*在本文中，我们设计并训练了一个生成式图像到文本的 Transformer 模型 GIT，以统一图像/视频字幕和问题回答等视觉-语言任务。虽然生成模型在预训练和微调之间提供了一致的网络架构，但现有工作通常包含复杂的结构（单/多模态编码器/解码器）并依赖于外部模块，例如对象检测器/标记器和光学字符识别（OCR）。在 GIT 中，我们将架构简化为一个图像编码器和一个文本解码器，属于单一的语言建模任务。我们还扩大了预训练数据和模型大小，以提高模型性能。毫不夸张地说，我们的 GIT 在 12 个具有挑战性的基准测试中以很大的优势创造了新的技术水平。例如，我们的模型在 TextCaps 上首次超越了人类表现（138.2 对 125.5 的 CIDEr）。此外，我们提出了一种新的基于生成的图像分类和场景文字识别方案，在标准基准测试中取得了不错的性能。*

提示：

- GIT 的实现方式与 GPT-2 非常相似，唯一的区别是该模型还以 `pixel_values` 为条件。

- 可以使用 [`GitProcessor`] 来准备模型的图像，并使用 `generate` 方法进行自回归生成。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/git_architecture.jpg"alt="drawing" width="600"/>
<small> GIT 架构。来自 <a href="https://arxiv.org/abs/2205.14100" target="_blank"> 原始论文 </a>。 </small>

该模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可在 [此处](https://github.com/microsoft/GenerativeImage2Text) 找到。

## 资源

以下是一些官方的 Hugging Face 资源和社区（由🌎表示），可帮助您入门 GIT。
- 有关在自定义数据上进行推理和微调 GIT 的演示笔记本可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/GIT) 找到。
- 另请参阅：[因果语言建模任务指南](../tasks/language_modeling)

如果您有兴趣提交资源以包含在这里，请随时提交拉取请求，我们将进行审查。该资源应该展示一些新的东西，而不是重复现有的资源。

## GitVisionConfig

[[autodoc]] GitVisionConfig

## GitVisionModel

[[autodoc]] GitVisionModel
    - forward

## GitConfig

[[autodoc]] GitConfig
    - all

## GitProcessor

[[autodoc]] GitProcessor
    - __call__

## GitModel

[[autodoc]] GitModel
    - forward

## GitForCausalLM

[[autodoc]] GitForCausalLM
    - forward
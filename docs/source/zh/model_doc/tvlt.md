<!--版权所有 2023 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证，第 2.0 版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下地址获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。--># TVLT
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TVLT

## 概述

TVLT 模型是由 Zineng Tang、Jaemin Cho、Yixin Nie、Mohit Bansal（前三位作者贡献相同）提出的 [《TVLT: Textless Vision-Language Transformer》](https://arxiv.org/abs/2209.14156)。TVLT（Textless Vision-Language Transformer）是一个使用原始视觉和音频输入进行视觉语言表示学习的模型，而无需使用文本特定的模块，如分词或自动语音识别（ASR）。它可以执行各种视觉语言和音频视觉任务，如检索、问答等。

论文中的摘要如下所示：

*在这项工作中，我们提出了 Textless Vision-Language Transformer（TVLT）, 其中同质 Transformer 模块接受原始视觉和音频输入进行视觉语言表示学习，最小化使用特定模态的设计，不使用分词或自动语音识别（ASR）等文本特定模块。TVLT 通过重建连续视频帧和音频频谱的掩码补丁（掩码自编码）和对齐视频和音频的对比建模进行训练。TVLT 在各种多模态任务上（如视觉问答、图像检索、视频检索和多模态情感分析）表现与以文本为基础的对应模型相当，具有 28 倍的推理速度和仅 1/3 的参数。我们的研究结果表明，即使在不假设文本先验存在的情况下，也有可能从低级别的视觉和音频信号中学习紧凑高效的视觉-语言表示。*

提示：

- TVLT 是一个将 `pixel_values` 和 `audio_values` 作为输入的模型。可以使用 [`TvltProcessor`](#tvltprocessor) 为该模型准备数据。  该处理器将图像处理器 (Image Processor)（用于图像/视频模态）和音频特征提取器（用于音频模态）封装在一起。
- TVLT 使用各种大小的图像/视频和音频进行训练：作者将输入的图像/视频调整大小并裁剪为 224，并限制音频频谱的长度为 2048。为了使视频和音频的批处理成为可能，作者使用 `pixel_mask` 表示哪些像素是真实/填充的，以及 `audio_mask` 表示哪些音频值是真实/填充的。
- TVLT 的设计与标准 Vision Transformer（ViT）和掩码自编码器（MAE）非常相似，就像 [ViTMAE](vitmae) 中一样。不同之处在于模型包括音频模态的嵌入层。
- 此模型的 PyTorch 版本仅适用于 torch 1.10 及更高版本。

<p align="center"> <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvlt_architecture.png"alt="drawing" width="600"/> </p>

<small> TVLT 架构。来自 <a href="[https://arxiv.org/abs/2102.03334](https://arxiv.org/abs/2209.14156)"> 原始论文 </a>。</small>

可在 [此处](https://github.com/zinengtang/TVLT) 找到原始代码。此模型由 [Zineng Tang](https://huggingface.co/ZinengTang) 贡献。

## TvltConfig

[[autodoc]] TvltConfig

## TvltProcessor

[[autodoc]] TvltProcessor
    - __call__

## TvltImageProcessor

[[autodoc]] TvltImageProcessor
    - preprocess

## TvltFeatureExtractor

[[autodoc]] TvltFeatureExtractor
    - __call__
    
## TvltModel

[[autodoc]] TvltModel
    - forward

## TvltForPreTraining

[[autodoc]] TvltForPreTraining
    - forward

## TvltForAudioVisualClassification

[[autodoc]] TvltForAudioVisualClassification
    - forward

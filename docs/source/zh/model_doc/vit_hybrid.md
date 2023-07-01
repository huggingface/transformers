<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证，版本 2.0（“许可证”）的规定，除非符合许可证的要求，否则您无权使用本文件。您可以在下面的链接处获得许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件在许可证下分发，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 混合视觉 Transformer（ViT Hybrid）
## 概述

混合视觉 Transformer（ViT）模型是由 Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、DirkWeissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、JakobUszkoreit、Neil Houlsby 在论文 [An Image is Worth 16x16 Words: Transformers for Image Recognitionat Scale](https://arxiv.org/abs/2010.11929) 中提出的。这是第一篇成功地在 ImageNet 上训练 Transformer 编码器并取得与常见的卷积架构相比非常好的结果的论文。ViT Hybrid 是 [plain Vision Transformer](vit) 的一个小变体，它利用了卷积主干（具体来说是 [BiT](bit)）的特征作为 Transformer 的初始“令牌”。

论文中的摘要如下所示：
*尽管 Transformer 架构已成为自然语言处理任务的事实标准，但其在计算机视觉领域的应用仍然有限。在视觉任务中，注意力要么与卷积网络一起使用，要么用来替换卷积网络的某些组件，同时保持其整体结构。我们表明，这种对卷积网络的依赖并非必需，直接应用纯 Transformer 在图像块序列上可以在图像分类任务上表现非常好。当在大量数据上进行预训练，并迁移到多个中小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）时，Vision Transformer（ViT）相比最先进的卷积网络取得了出色的结果，同时所需的计算资源要少得多。* 
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码（使用 JAX 编写）可以在
此处找到（https://github.com/google-research/vision_transformer）。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 标示）资源的列表，可帮助您开始使用 ViT Hybrid。
<PipelineTag pipeline="image-classification"/>
- [`ViTHybridForImageClassification`] 在此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 中提供支持。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将对其进行审查！该资源应该展示出新的内容，而不是重复现有的资源。


## ViTHybridConfig
[[autodoc]] ViTHybridConfig
## ViTHybridImageProcessor
[[autodoc]] ViTHybridImageProcessor    - preprocess
## ViTHybridModel
[[autodoc]] ViTHybridModel    - forward
## ViTHybridForImageClassification
[[autodoc]] ViTHybridForImageClassification    - forward
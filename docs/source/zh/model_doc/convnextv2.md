<!--版权所有 2023 年的 HuggingFace 团队。保留所有权利。
按照 Apache 许可证第 2.0 版（"许可证"）许可; 除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于 "按原样" 的基础分发的，没有任何形式的担保或条件，无论是明示还是暗示。有关许可证的特定语言下的权限和限制的详细信息，请参阅许可证。
⚠️请注意，此文件是 Markdown 格式的，但包含我们的 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# ConvNeXt V2


## 概述

ConvNeXt V2 模型是由 Sanghyun Woo、Shoubhik Debnath、Ronghang Hu、Xinlei Chen、Zhuang Liu、In So Kweon 和 Saining Xie 在 [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) 中提出的。ConvNeXt V2 是一个纯卷积模型（ConvNet），灵感来自 Vision Transformers 的设计，是 [ConvNeXT](convnext) 的继任者。

来自论文的摘要如下:
*在改进的架构和更好的表示学习框架的推动下，视觉识别领域在 2020 年初取得了快速的现代化和性能提升。例如，现代的 ConvNets，以 ConvNeXt 为代表，已经在各种场景中展示了强大的性能。虽然这些模型最初是为带有 ImageNet 标签的监督学习而设计的，但它们也有可能受益于诸如 masked autoencoders (MAE)等自监督学习技术。然而，我们发现简单地将这两种方法结合起来会导致性能下降。在本文中，我们提出了一个完全卷积的 masked autoencoder 框架和一个新的全局响应归一化（GRN）层，可以添加到 ConvNeXt 架构中以增强通道间特征的竞争。这种自监督学习技术和架构改进的协同设计产生了一个名为 ConvNeXt V2 的新模型系列，显著提高了纯 ConvNets 在各种识别基准上的性能，包括 ImageNet 分类、COCO 检测和 ADE20K 分割。我们还提供了各种大小的预训练 ConvNeXt V2 模型，从高效的 3.7M 参数 Atto 模型（在 ImageNet 上具有 76.7%的 top-1 准确率）到 650M 的巨型模型（仅使用公共训练数据实现了最新的 88.9%准确率）.*

提示:

- 有关用法，请参阅每个模型下方的代码示例。
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png"alt="drawing" width="600"/>

<small> ConvNeXt V2 架构。来自 <a href="https://arxiv.org/abs/2301.00808"> 原始论文 </a>。</small>

此模型由 [adirik](https://huggingface.co/adirik) 贡献。原始代码可以在 [这里](https://github.com/facebookresearch/ConvNeXt-V2) 找到。

## 资源

官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 ConvNeXt V2。

<PipelineTag pipeline="image-classification"/>
- [`ConvNextV2ForImageClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持。

如果您有兴趣提交资源以便收录在此处，请随时提出拉取请求，我们将进行审核！该资源应该展示出一些新的东西，而不是重复现有的资源。

## ConvNextV2Config

[[autodoc]] ConvNextV2Config

## ConvNextV2Model

[[autodoc]] ConvNextV2Model
    - forward
## ConvNextV2ForImageClassification

[[autodoc]] ConvNextV2ForImageClassification
    - forward
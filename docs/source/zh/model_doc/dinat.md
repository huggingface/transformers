<!--版权所有2022年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；您不得使用此文件，除非符合许可证。您可以在以下网址获取许可证副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证了解特定语言下许可权限和限制。
⚠️ 注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能无法在 Markdown 查看器中正确呈现。
-->

# 扩张邻域注意力变换器 (Dilated Neighborhood Attention Transformer)

## 概述

DiNAT 是由 Ali Hassani 和 Humphrey Shi 在 [扩张邻域注意力变换器](https://arxiv.org/abs/2209.15001) 中提出的扩展了 [NAT](nat) 的模型。它通过添加扩张邻域注意力模式来捕捉全局上下文，并且在性能上显示出明显的改进。
论文中的摘要如下所示：

*变换器正在迅速成为跨模态，领域和任务中应用最广泛的深度学习架构之一。在视觉领域，除了朴素变换器的持续努力外，分层变换器也因其性能和易于集成到现有框架中而受到广泛关注。这些模型通常使用局部注意机制，例如滑动窗口邻域注意力（NA）或 Swin 变换器的偏移窗口自注意力。虽然这些机制可以有效降低自注意力的二次复杂度，但局部注意力削弱了自注意力的两个最有吸引力的特性：长程相互依赖建模和全局感受野。在本文中，我们引入了扩张邻域注意力（DiNA），它是 NA 的一种自然，灵活且高效的扩展，可以在不增加额外成本的情况下捕捉更多的全局上下文并以指数级扩展感受野。NA 的局部关注和 DiNA 的稀疏全局关注相互补充，因此我们引入了扩张邻域注意力变换器（DiNAT），这是一个新的分层视觉变换器，它结合了两者。DiNAT 变体在 NAT、Swin 和 ConvNeXt 等强基线模型上实现了显著的改进。我们的大型模型在 COCO 目标检测中比其 Swin 对应模型提前 1.5%的框 AP，在 COCO 实例分割中提前 1.3%的掩模 AP，在 ADE20K 语义分割中提前 1.1%的 mIoU 与新框架搭配使用，我们的大型变体成为 COCO（58.2 PQ）和 ADE20K（48.5 PQ）的新一代全景分割模型，Cityscapes（44.5 AP）和 ADE20K（35.4 AP）的实例分割模型（无额外数据）。它还与 ADE20K（58.2 mIoU）的最先进的专用语义分割模型相匹配，并在 Cityscapes（84.5 mIoU）上排名第二（无额外数据）。* and ranks second on Cityscapes (84.5 mIoU) (no extra data). *

提示：
- 您可以使用 [`AutoImageProcessor`] API 为模型准备图像。
- DiNAT 可用作 *骨干*。当 `output_hidden_states = True` 时，它将输出 `hidden_states` 和 `reshaped_hidden_states`。`reshaped_hidden_states` 的形状为 `(batch, num_channels, height, width)`，而不是 `(batch_size, height, width, num_channels)`。

注：
- DiNAT 依赖于 [NATTEN](https://github.com/SHI-Labs/NATTEN/) 对邻域注意力和扩张邻域注意力的实现。您可以通过参考 [shi-labs.com/natten](https://shi-labs.com/natten) 获取 Linux 的预构建轮子进行安装，或者通过运行 `pip install natten` 在系统上进行构建。请注意，后者可能需要一些时间进行编译。NATTEN 尚不支持 Windows 设备。
- 目前仅支持 4 个补丁大小。

<imgsrc="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dilated-neighborhood-attention-pattern.jpg"alt="drawing" width="600"/>

<small> 不同扩张值的邻域注意力。摘自 <a href="https://arxiv.org/abs/2209.15001"> 原始论文 </a>。</small>

此模型由 [Ali Hassani](https://huggingface.co/alihassanijr) 贡献。原始代码可以在 [此处](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) 找到。

## 资源
以下是一些官方 Hugging Face 和社区（通过🌎表示）的资源，可以帮助您开始使用 DiNAT。

<PipelineTag pipeline="image-classification"/>

- 通过此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持使用 [`DinatForImageClassification`]。

如果您有兴趣提交要包含在此处的资源，请随时发起拉取请求，我们将进行审核！资源应该展示一些新东西，而不是重复现有的资源。

## DinatConfig

[[autodoc]] DinatConfig

## DinatModel

[[autodoc]] DinatModel
    - forward

## DinatForImageClassification

[[autodoc]] DinatForImageClassification
    - forward
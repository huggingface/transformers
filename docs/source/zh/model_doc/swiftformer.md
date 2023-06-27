<!--版权所有2023年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法
在您的 Markdown 查看器中正确渲染。-->


# SwiftFormer

## 概述

SwiftFormer 模型在 [SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://arxiv.org/abs/2303.15446) 一文中由 Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, Fahad Shahbaz Khan 提出。
SwiftFormer 论文引入了一种新颖的高效加性注意力机制，通过线性逐元素乘法有效地替代自注意力计算中的二次矩阵乘法运算。基于此构建了一系列名为'SwiftFormer'的模型，其在准确性和移动推理速度方面均达到了最先进的性能。即使是其小型变种，在 iPhone 14 上只有 0.8 毫秒的延迟，准确率达到了 78.5 ％的 ImageNet1K top-1 准确率，比 MobileViT-v2 更准确且快 2 倍。
论文中的摘要如下：
*自注意力已成为各种视觉应用中捕捉全局上下文的事实选择。然而，其与图像分辨率相关的二次计算复杂度限制了其在实时应用中的使用，特别是在资源受限的移动设备上部署。虽然已经提出了混合方法，以在速度和准确性之间取得更好的平衡，但是自注意力中昂贵的矩阵乘法操作仍然是一个瓶颈。在本文中，我们引入了一种新颖的高效加性注意力机制，通过线性逐元素乘法有效地替代了二次矩阵乘法运算。我们的设计表明，可以通过线性层来替换键值交互而不会损失任何准确性。与以往最先进的方法不同，我们对自注意力的高效构造使其在网络的所有阶段都可以使用。使用我们提出的高效加性注意力，我们构建了一系列名为'SwiftFormer'的模型，其在准确性和移动推理速度方面均达到了最先进的性能。我们的小型变种在 iPhone 14 上只有 0.8 毫秒的延迟，准确率达到了 78.5 ％的 ImageNet-1K top-1 准确率，比 MobileViT-v2 更准确且快 2 倍。*
提示：    

- 您可以使用 [`ViTImageProcessor`] API 为模型准备图片。

此模型由 [shehan97](https://huggingface.co/shehan97) 贡献。原始代码可以在 [此处](https://github.com/Amshaker/SwiftFormer) 找到。

## SwiftFormerConfig
[[autodoc]] SwiftFormerConfig
## SwiftFormerModel
[[autodoc]] SwiftFormerModel    - forward
## SwiftFormerForImageClassification
[[autodoc]] SwiftFormerForImageClassification    - forward
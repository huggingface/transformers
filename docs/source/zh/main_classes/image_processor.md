<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 文件，但包含我们的文档构建器（类似 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 图像处理器 (Image Processor)

图像处理器 (Image Processor)负责为视觉模型准备输入特征并对其输出进行后处理。这包括诸如调整大小、归一化和转换为 PyTorch、TensorFlow、Flax 和 Numpy 张量的转换。它还可以包括模型特定的后处理，例如将 logits 转换为分割掩码。

## ImageProcessingMixin

[[autodoc]] image_processing_utils.ImageProcessingMixin
    - from_pretrained
    - save_pretrained

## BatchFeature

[[autodoc]] BatchFeature

## BaseImageProcessor

[[autodoc]] image_processi
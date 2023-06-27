<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何明示或暗示的保证或条件。请参阅许可证以了解具体语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。特定语言下的许可证。
-->



# Utilities for Image Processors

此页面列出了图像处理器 (Image Processor)使用的所有实用函数，主要用于处理图像的功能性转换。这些大多数函数只在您学习库中的图像处理器 (Image Processor)代码时才有用。


## 图像转换

[[autodoc]] image_transforms.center_crop
[[autodoc]] image_transforms.center_to_corners_format
[[autodoc]] image_transforms.corners_to_center_format
[[autodoc]] image_transforms.id_to_rgb
[[autodoc]] image_transforms.normalize
[[autodoc]] image_transforms.pad
[[autodoc]] image_transforms.rgb_to_id
[[autodoc]] image_transforms.rescale
[[autodoc]] image_transforms.resize
[[autodoc]] image_transforms.to_pil_image
## ImageProcessingMixin
[[autodoc]] image_processing_utils.ImageProcessingMixin
<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在许可证的以下网址获取许可证副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了特定于我们的文档构建器（类似于 MDX）的语法，您的 Markdown 查看器可能无法正确渲染。
-->

# CLIPSeg（模型名称）

## 概述

CLIPSeg 模型是由 Timo L ü ddecke 和 Alexander Ecker 在《使用文本和图像提示进行图像分割》（https://arxiv.org/abs/2112.10003）中提出的。CLIPSeg 在冻结的 [CLIP](clip) 模型的基础上添加了一个最小的解码器，用于零样本和一样本图像分割。论文中的摘要如下：

*图像分割通常通过为一组固定的对象类别训练模型来解决。随后，将额外的类别或更复杂的查询合并到模型中是昂贵的，因为这需要在涵盖这些表达的数据集上重新训练模型。在这里，我们提出了一种可以根据任意提示在测试时生成图像分割的系统。提示可以是文本或图像。这种方法使我们能够为三个常见的分割任务创建一个统一的模型（仅训练一次），这些任务具有不同的挑战：指代表达式分割、零样本分割和一样本分割。我们在 CLIP 模型的基础上进行扩展，使用基于 Transformer 的解码器实现稠密预测。在扩展的 PhraseCut 数据集上训练后，我们的系统可以根据自由文本提示或额外的图像来生成图像的二进制分割图。我们详细分析了后一种基于图像的提示的不同变体。这种新颖的混合输入不仅允许动态适应上述三个分割任务，还可以应用于任何文本或图像查询可表达为二进制分割任务的情况。最后，我们发现我们的系统很好地适应了涉及功能或属性的广义查询*

论文中的摘要如下：
- [CLIPSegForImageSegmentation] 在 [CLIPSegModel] 之上添加了一个解码器。后者与 [CLIPModel] 相同。
- [CLIPSegForImageSegmentation] 可以根据任意提示在测试时生成图像分割。提示可以是文本（作为“input_ids”提供给模型）或图像（作为“conditional_pixel_values”提供给模型）。还可以提供自定义条件嵌入（作为“conditional_embeddings”提供给模型）。

< img src =" https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/clipseg_architecture.png "
alt = "drawing" width = "600"/> 

<small> CLIPSeg 概述。摘自 <a href="https://arxiv.org/abs/2112.10003"> 原始论文。</a> </small>

此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在 [这里](https://github.com/timojl/clipseg) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源的列表，可帮助您开始使用 CLIPSeg。如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将进行审核！该资源理想情况下应该展示一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="image-segmentation"/>

- 一个笔记本，演示了 [使用 CLIPSeg 进行零样本图像分割](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb)。

## CLIPSegConfig

[[autodoc]] CLIPSegConfig
    - from_text_vision_configs

## CLIPSegTextConfig

[[autodoc]] CLIPSegTextConfig

## CLIPSegVisionConfig

[[autodoc]] CLIPSegVisionConfig

## CLIPSegProcessor

[[autodoc]] CLIPSegProcessor

## CLIPSegModel

[[autodoc]] CLIPSegModel
    - forward
    - get_text_features
    - get_image_features

## CLIPSegTextModel

[[autodoc]] CLIPSegTextModel
    - forward

## CLIPSegVisionModel

[[autodoc]] CLIPSegVisionModel
    - forward

## CLIPSegForImageSegmentation

[[autodoc]] CLIPSegForImageSegmentation
    - forward
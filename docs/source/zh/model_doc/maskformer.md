<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发，不附带任何形式的担保或条件。请参阅许可证以了解具体的语言权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->

# MaskFormer

<Tip>

这是一个最近推出的模型，所以API尚未经过广泛测试。可能会有一些错误或稍微的变化，需要在将来进行修复。如果遇到任何奇怪的问题，请提交[Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

MaskFormer模型是由Bowen Cheng、Alexander G. Schwing和Alexander Kirillov在论文[Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278)中提出的。MaskFormer通过使用掩码分类范式而不是传统的像素级分类来处理语义分割问题。

论文的摘要如下：

*现代方法通常将语义分割视为像素级分类任务，而实例级分割则使用替代的掩码分类方法处理。我们的关键洞察是：掩码分类足够通用，可以使用完全相同的模型、损失和训练过程来统一解决语义分割和实例级分割任务。基于这一观察，我们提出了MaskFormer，一个简单的掩码分类模型，它预测一组二进制掩码，每个掩码与单个全局类标签预测相关联。总体而言，基于掩码分类的方法简化了语义分割和全景分割任务的有效方法，展现了出色的实证结果。特别是，在类别数量较多时，我们观察到MaskFormer优于像素级分类基准。我们的基于掩码分类的方法优于当前最先进的语义分割（ADE20K上的55.6 mIoU）和全景分割（COCO上的52.7 PQ）模型。*

提示:

- MaskFormer的Transformer解码器与[DETR](detr)的解码器完全相同。在训练过程中，DETR的作者发现在解码器中使用辅助损失非常有帮助，特别是帮助模型输出每个类别的正确对象数量。如果将[`MaskFormerConfig`]的`use_auxilary_loss`参数设置为`True`，则在每个解码器层之后添加预测前馈神经网络和匈牙利损失（其中FFNs共享参数）。

- 如果要在多节点的分布式环境中训练模型，则需要更新`modeling_maskformer.py`中`MaskFormerLoss`类中的`get_num_masks`函数。在多节点训练时，应将其设置为所有节点上目标掩码的平均数量，原始实现可参考[这里](https://github.com/facebookresearch/MaskFormer/blob/da3e60d85fdeedcb31476b5edd7d328826ce56cc/mask_former/modeling/criterion.py#L169)。
- 可以使用[`MaskFormerImageProcessor`]为模型准备图像数据以及可选的目标数据。

- 根据任务需求，可以调用[`~MaskFormerImageProcessor.post_process_semantic_segmentation`]或[`~MaskFormerImageProcessor.post_process_panoptic_segmentation`]来获取最终的分割结果。这两个任务都可以使用[`MaskFormerForInstanceSegmentation`]的输出进行解决，其中panoptic分割可以使用可选的`label_ids_to_fuse`参数将目标对象（如天空）的实例进行融合。

下图展示了MaskFormer的架构，摘自[原始论文](https://arxiv.org/abs/2107.06278)。

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png"/>

该模型由[francesco](https://huggingface.co/francesco)贡献。原始代码可以在[这里](https://github.com/facebookresearch/MaskFormer)找到。

## Resources

<PipelineTag pipeline="image-segmentation"/>

- 所有展示MaskFormer的推理以及在自定义数据上进行微调的notebook可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer)找到。

## MaskFormer specific outputs

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerModelOutput

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput

## MaskFormerConfig

[[autodoc]] MaskFormerConfig

## MaskFormerImageProcessor

[[autodoc]] MaskFormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerFeatureExtractor

[[autodoc]] MaskFormerFeatureExtractor
    - __call__
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerModel

[[autodoc]] MaskFormerModel
    - forward

## MaskFormerForInstanceSegmentation

[[autodoc]] MaskFormerForInstanceSegmentation
    - forward

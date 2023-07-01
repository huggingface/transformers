<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”方式分发，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->
# OneFormer

## 概述

OneFormer 模型由 Jitesh Jain、Jiachen Li、MangTik Chiu、Ali Hassani、Nikita Orlov 和 Humphrey Shi 在 [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) 中提出。

OneFormer 是一个通用的图像分割框架，可以在单个全景数据集上进行训练，执行语义、实例和全景分割任务。OneFormer 使用任务令牌来使模型根据当前任务进行条件训练，并使体系结构在推断中动态适应任务。

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_teaser.png"/>
论文中的摘要如下：

*通用图像分割并不是一个新概念。过去几十年来，统一图像分割的尝试包括场景解析、全景分割，以及最近的新全景架构。然而，这些全景架构并不能真正统一图像分割，因为它们需要在语义、实例或全景分割上分别进行训练，以达到最佳性能。理想情况下，一个真正通用的框架只需要训练一次，并在所有三个图像分割任务上实现 SOTA 性能。为此，我们提出了 OneFormer，这是一个通用的图像分割框架，通过多任务训练设计将分割统一起来。首先，我们提出了一种任务条件的联合训练策略，使得在单个多任务训练过程中能够训练每个领域（语义、实例和全景分割）的真值。其次，我们引入了一个任务令牌，将我们的模型与正在处理的任务进行关联，使我们的模型能够支持多任务训练和推断。第三，我们在训练过程中提出了一种查询-文本对比损失，以建立更好的跨任务和跨类别区分。值得注意的是，我们的单个 OneFormer 模型在 ADE20k、CityScapes 和 COCO 上的所有三个分割任务上表现优于专门的 Mask2Former 模型，尽管后者在每个任务上分别使用了三倍的资源进行训练。通过使用新的 ConvNeXt 和 DiNAT 骨干，我们观察到了更多的性能改进。我们相信 OneFormer 是使图像分割更具普适性和可访问性的重要一步。*

提示：
- OneFormer 在推断过程中需要两个输入：*image* 和 *task token*。
- 在训练过程中，OneFormer 仅使用全景标注。
- 如果您要在多个节点的分布式环境中训练模型，则应在 `modeling_oneformer.py` 中的 `OneFormerLoss` 类的 `get_num_masks` 函数中进行更新。在多个节点上训练时，应将其设置为所有节点上目标掩码的平均数，可以在原始实现 [此处](https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/criterion.py#L287) 中看到。
- 您可以使用 [`OneFormerProcessor`] 来准备模型的输入图像和任务输入，以及模型的可选目标。[`OneFormerProcessor`] 将 [`OneFormerImageProcessor`] 和 [`CLIPTokenizer`] 封装到单个实例中，用于同时准备图像和编码任务输入。
- 要获取最终的分割结果，根据任务的不同，可以调用 [`~OneFormerProcessor.post_process_semantic_segmentation`] 或 [`~OneFormerImageProcessor.post_process_instance_segmentation`] 或 [`~OneFormerImageProcessor.post_process_panoptic_segmentation`]。这三个任务都可以使用 [`OneFormerForUniversalSegmentation`] 的输出解决，全景分割接受一个可选的 `label_ids_to_fuse` 参数，用于将目标对象（例如天空）的实例合并在一起。
下图展示了 OneFormer 的体系结构。摘自 [原始论文](https://arxiv.org/abs/2211.06220)。
<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png"/>
此模型由 [Jitesh Jain](https://huggingface.co/praeclarumjj3) 贡献。原始代码可以在 [此处](https://github.com/SHI-Labs/OneFormer) 找到。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 表示）资源列表，可帮助您开始使用 OneFormer。

- 可在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OneFormer) 找到有关推断 + 自定义数据微调的演示笔记本。
如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将进行审查。该资源应该展示出一些新东西，而不是重复现有的资源。

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerModelOutput

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput

## OneFormerConfig

[[autodoc]] OneFormerConfig

## OneFormerImageProcessor

[[autodoc]] OneFormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## OneFormerProcessor

[[autodoc]] OneFormerProcessor

## OneFormerModel

[[autodoc]] OneFormerModel
    - forward

## OneFormerForUniversalSegmentation

[[autodoc]] OneFormerForUniversalSegmentation
    - forward
    
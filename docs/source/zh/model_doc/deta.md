<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用的法律要求或书面同意，根据许可证分发的软件是按“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解具体语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，您的 Markdown 查看器可能无法正确渲染。渲染。
-->
# DETA

## 概述

DETA 模型是由 Jeffrey Ouyang-Zhang，Jang Hyun Cho，Xingyi Zhou，Philipp Kr ä henb ü hl 在 [NMS Strikes Back](https://arxiv.org/abs/2212.06137) 中提出的。DETA（Detection Transformers with Assignment 的简称）通过将一对一的二部图匹配损失替换为传统检测器中使用的一对多标签分配以及非极大值抑制（NMS），从而改进了 [Deformable DETR](deformable_detr)。这导致了高达 2.5 mAP 的显著增益。来自论文的摘要如下：

*目标检测器（DETR）通过在训练过程中使用一对一的二部图匹配直接将查询转换为唯一对象，并实现端到端目标检测。最近，这些模型在 COCO 上以不可否认的优雅超过了传统检测器。然而，它们与传统检测器在多个设计方面存在差异，包括模型架构和训练计划，因此一对一匹配的有效性尚未完全理解。在这项工作中，我们在 DETR 的一对一匈牙利匹配和传统检测器中的一对多标签分配与非极大值抑制（NMS）之间进行了严格比较。令人惊讶的是，在相同设置下，我们观察到 NMS 中的一对多分配始终优于标准的一对一匹配，显著提高了高达 2.5 mAP。我们的检测器使用传统的基于 IoU 的标签分配训练 Deformable-DETR，在 12 个 epochs（1x 计划）和 ResNet50 骨干网络下实现了 50.2 COCO mAP，在这种设置下，性能优于所有现有的传统或基于 Transformer 的检测器。在多个数据集、计划和架构上，我们一直在展示二部图匹配对于高性能的目标检测器是不必要的。此外，我们将目标检测器的成功归因于其具有表达力的 Transformer 架构。*

提示：

- 您可以使用 [`DetaImageProcessor`] 为模型准备图像和可选的目标。
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deta_architecture.jpg"alt="drawing" width="600"/>
<small> DETA 概览。

来自 <a href="https://arxiv.org/abs/2212.06137"> 原始论文 </a>。</small>
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。

原始代码可以在 [此处](https://github.com/jozhang97/DETA) 找到。
## 资源
官方 Hugging Face 和社区（使用🌎标记）的资源列表，可帮助您开始使用 DETA。

- DETA 的演示笔记本可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA) 找到。
- 另请参阅：[目标检测任务指南](../tasks/object_detection)

如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将进行审查！资源应理想地展示新的东西，而不是重复现有的资源。

## DetaConfig

[[autodoc]] DetaConfig


## DetaImageProcessor

[[autodoc]] DetaImageProcessor
    - preprocess
    - post_process_object_detection


## DetaModel

[[autodoc]] DetaModel
    - forward


## DetaForObjectDetection

[[autodoc]] DetaForObjectDetection
    - forward

<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在下面获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样” BASIS，不提供任何明示或暗示的担保或条件。请参阅许可证特定语言的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确渲染。
-->
# DPT

## 概述

DPT 模型是由 Ren é Ranftl、Alexey Bochkovskiy、Vladlen Koltun 于 [《Vision Transformers for Dense Prediction》](https://arxiv.org/abs/2103.13413) 中提出的。DPT 是一种利用 [视觉 Transformer（ViT）](vit) 作为密集预测任务（如语义分割和深度估计）的骨干的模型。

论文中的摘要如下：

*我们引入了密集视觉 Transformer，这是一种将视觉 Transformer 代替卷积网络作为密集预测任务骨干的体系结构。我们从视觉 Transformer 的各个阶段汇集标记，形成各种分辨率的类似图像的表示，并使用卷积解码器逐步将它们组合成完整分辨率的预测。Transformer 骨干在恒定且相对较高的分辨率下处理表示，并在每个阶段具有全局感受野。这些属性使得密集视觉 Transformer 相比完全卷积网络能够提供更细粒度和更全局一致的预测。我们的实验证明，该体系结构在密集预测任务上取得了显著的改进，特别是在拥有大量训练数据时。对于单目深度估计，与最先进的完全卷积网络相比，我们观察到相对性能提升高达 28%。在语义分割方面，密集视觉 Transformer 在 ADE20K 上达到了 49.02%的 mIoU，创造了新的最佳性能。我们还展示了该体系结构可以在更小的数据集（如 NYUv2、KITTI 和 Pascal Context）上进行微调，在这些数据集上也创造了新的最佳性能。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg"alt="drawing" width="600"/>
<small> DPT 架构。

取自 <a href="https://arxiv.org/abs/2103.13413" target="_blank"> 原始论文 </a>。 </small>
该模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可在 [此处](https://github.com/isl-org/DPT) 找到。

## 资源

官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您入门 DPT。

- [`DPTForDepthEstimation`] 的演示笔记本可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DPT) 找到。
- [语义分割任务指南](../tasks/semantic_segmentation)- [单目深度估计任务指南](../tasks/monocular_depth_estimation)
如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将对其进行审查！该资源应该展示出一些新的东西，而不是重复现有资源。

## DPTConfig

[[autodoc]] DPTConfig

## DPTFeatureExtractor

[[autodoc]] DPTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## DPTImageProcessor

[[autodoc]] DPTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## DPTModel

[[autodoc]] DPTModel
    - forward

## DPTForDepthEstimation

[[autodoc]] DPTForDepthEstimation
    - forward

## DPTForSemanticSegmentation

[[autodoc]] DPTForSemanticSegmentation
    - forward

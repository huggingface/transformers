<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不能使用此文件。您可以在下面获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，依照许可证分发的软件是按照“原样”基础分发的，不附带任何形式的担保或条件。请阅读许可证以了解特定语言下的权限和限制。
⚠️ 请注意，该文件是 Markdown 格式，但包含我们文档构建器的特定语法（类似于 MDX），可能在您的 Markdown 查看器中无法正常呈现。
-->
# VideoMAE

## 概述

VideoMAE 模型是由 Zhan Tong、Yibing Song、Jue Wang、Limin Wang 在 [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) 中提出的。VideoMAE 将掩码自编码器（MAE）扩展到视频领域，在几个视频分类基准上声称具有最先进的性能。

论文摘要如下：

*在相对较小的数据集上实现卓越性能通常需要在超大规模数据集上预训练视频变换器。在本文中，我们展示了视频掩码自编码器（VideoMAE）是用于自监督视频预训练（SSVP）的数据高效学习器。我们受到最近的 ImageMAE 的启发，并提出了定制的视频管掩蔽和重构。这些简单的设计证明在克服视频重建期间由时间相关性引起的信息泄漏方面非常有效。我们对 SSVP 获得了三个重要发现：（1）极高比例的掩蔽比率（即 90%至 95%）仍然可以产生有利的 VideoMAE 性能。与图像相比，时间上冗余的视频内容使得掩蔽比率更高。（2）VideoMAE 在非常小的数据集上（即约 3k-4k 个视频）取得了令人印象深刻的结果，而不使用任何额外的数据。这部分归因于视频重建任务的挑战，以实现高级结构学习。（3）VideoMAE 表明对于 SSVP，数据质量比数据数量更重要。预训练和目标数据集之间的领域转移是 SSVP 的重要问题。值得注意的是，我们的 VideoMAE 与普通的 ViT 骨干可以在 Kinects-400 上实现 83.9 ％，在 Something-Something V2 上实现 75.3 ％，在 UCF101 上实现 90.8 ％，在 HMDB51 上实现 61.1 ％，而不使用任何额外的数据。*

提示：

- 您可以使用 [`VideoMAEImageProcessor`] 为模型准备视频。它将为您调整大小+归一化视频的所有帧。- [`VideoMAEForPreTraining`] 包含自监督预训练的解码器。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/videomae_architecture.jpeg"alt="drawing" width="600"/>
<small> VideoMAE 预训练。

摘自 <a href="https://arxiv.org/abs/2203.12602"> 原始论文 </a>。</small>
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在 [这里](https://github.com/MCG-NJU/VideoMAE) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源的列表，可帮助您开始使用 VideoMAE。如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将进行审核！资源应该展示一些新内容，而不是重复现有的资源。

**视频分类**
- [一个笔记本](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb)，展示如何对自定义数据集进行微调 VideoMAE 模型。
- [视频分类任务指南](../tasks/video-classification)
- [一个🤗空间](https://huggingface.co/spaces/sayakpaul/video-classification-ucf101-subset)，展示如何使用视频分类模型进行推断。


## VideoMAEConfig

[[autodoc]] VideoMAEConfig

## VideoMAEFeatureExtractor

[[autodoc]] VideoMAEFeatureExtractor
    - __call__

## VideoMAEImageProcessor

[[autodoc]] VideoMAEImageProcessor
    - preprocess

## VideoMAEModel

[[autodoc]] VideoMAEModel
    - forward

## VideoMAEForPreTraining

[[autodoc]] transformers.VideoMAEForPreTraining
    - forward

## VideoMAEForVideoClassification

[[autodoc]] transformers.VideoMAEForVideoClassification
    - forward

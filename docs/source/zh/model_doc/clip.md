<!--版权所有2021年HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用的法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档构建器的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。
-->

# CLIP

## 概述

CLIP 模型是由 Alec Radford，Jong Wook Kim，Chris Hallacy，Aditya Ramesh，Gabriel Goh，Sandhini Agarwal，Girish Sastry，Amanda Askell，Pamela Mishkin，Jack Clark，Gretchen Krueger，Ilya Sutskever 在 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) 中提出的。CLIP（对比语言-图像预训练）是一个在各种（图像，文本）对上进行训练的神经网络。它可以根据自然语言的指令，在给定图像的情况下预测最相关的文本片段，而无需直接针对任务进行优化，类似于 GPT-2 和 3 的零-shot 能力。

来自论文的摘要如下：

*最先进的计算机视觉系统被训练以预测一组固定的预定义对象类别。这种受限形式的监督限制了它们的普适性和可用性，因为需要额外的已标记数据来指定任何其他视觉概念。直接从原始文本中学习关于图像的内容是一种有前途的替代方法，它利用了更广泛的监督来源。我们证明，预测哪个标题与哪个图像配对的简单预训练任务是一种高效且可扩展的从头开始学习 SOTA 图像表示的方法，该方法使用了从互联网收集的 4 亿个（图像，文本）对的数据集。在预训练之后，自然语言用于引用已学习的视觉概念（或描述新的概念），从而使模型能够零-shot 地迁移到下游任务。我们通过对超过 30 个不同的现有计算机视觉数据集进行基准测试来研究此方法的性能，这些数据集涵盖了 OCR，视频中的动作识别，地理定位以及许多类型的细粒度对象分类等任务。该模型在大多数任务中都能够进行非平凡的迁移，并且通常能够与完全监督的基线模型竞争，而无需进行任何特定于数据集的训练。例如，我们在 ImageNet 零-shot 任务上匹配了原始 ResNet-50 的准确度，而无需使用其训练的 128 万个样本之一。我们在此 https URL 上发布了我们的代码和预训练模型权重。*

## 用途 

CLIP 是一种多模态视觉和语言模型。它可以用于图像-文本相似度和零-shot 图像分类。CLIP 使用类似 ViT 的 Transformer 获取视觉特征，并使用因果语言模型获取文本特征。然后，文本和视觉特征都被投影到具有相同维度的潜空间中。投影后的图像和文本特征之间的点积被用作相似分数。

为了将图像馈送到 Transformer 编码器中，每个图像被分割成一系列固定大小且不重叠的补丁，然后进行线性嵌入。添加一个 [CLS] 令牌作为整个图像的表示。作者还添加了绝对位置嵌入，并将所得的向量序列馈送到标准 Transformer 编码器中。

[`CLIPFeatureExtractor`] 可用于为模型调整大小（或重新缩放）和归一化图像。
[`CLIPTokenizer`] 用于对文本进行编码。[`CLIPProcessor`] 将 [`CLIPFeatureExtractor`] 和 [`CLIPTokenizer`] 封装为单个实例，既可以编码文本，又可以准备图像。以下示例展示了如何使用 [`CLIPProcessor`] 和 [`CLIPModel`] 获取图像-文本相似度分数。

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

此模型由 [valhalla](https://huggingface.co/valhalla) 贡献。原始代码可以在此处找到：[链接](https://github.com/openai/CLIP)。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 CLIP。

- 一篇关于 [如何在 10,000 个图像-文本对上微调 CLIP](https://huggingface.co/blog/fine-tune-clip-rsicd) 的博客文章。
- CLIP 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text) 支持。

如果您有兴趣提交资源以包含在此处，请随时发起 Pull Request，我们将对其进行审核。资源应该展示一些新东西，而不是重复现有的内容。
## CLIPConfig

[[autodoc]] CLIPConfig
    - from_text_vision_configs

## CLIPTextConfig

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer

[[autodoc]] CLIPTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CLIPTokenizerFast

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPFeatureExtractor

[[autodoc]] CLIPFeatureExtractor

## CLIPProcessor

[[autodoc]] CLIPProcessor

## CLIPModel

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection

[[autodoc]] CLIPVisionModelWithProjection
    - forward


## CLIPVisionModel

[[autodoc]] CLIPVisionModel
    - forward

## TFCLIPModel

[[autodoc]] TFCLIPModel
    - call
    - get_text_features
    - get_image_features

## TFCLIPTextModel

[[autodoc]] TFCLIPTextModel
    - call

## TFCLIPVisionModel

[[autodoc]] TFCLIPVisionModel
    - call

## FlaxCLIPModel

[[autodoc]] FlaxCLIPModel
    - __call__
    - get_text_features
    - get_image_features

## FlaxCLIPTextModel

[[autodoc]] FlaxCLIPTextModel
    - __call__

## FlaxCLIPVisionModel

[[autodoc]] FlaxCLIPVisionModel
    - __call__

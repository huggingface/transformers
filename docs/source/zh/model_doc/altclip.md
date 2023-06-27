<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS，不附带任何形式的担保或条件。有关许可证的特定语言的权限和限制，请参阅许可证。
⚠️请注意，此文件是 Markdown 格式，但包含了我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# AltCLIP

## 概述

AltCLIP 模型是由 Zhongzhi Chen、Guang Liu、Bo-Wen Zhang、Fulong Ye、Qinghong Yang 和 Ledell Wu 在 [AltCLIP：改变 CLIP 中的语言编码器以扩展语言功能](https://arxiv.org/abs/2211.06679v2) 一文中提出的。AltCLIP（改变 CLIP 中的语言编码器）是一种在各种图像-文本和文本-文本对上进行训练的神经网络。通过将 CLIP 的文本编码器切换为预训练的多语言文本编码器 XLM-R，我们可以在几乎所有任务上获得与 CLIP 非常接近的性能，并扩展原始 CLIP 的功能，如多语言理解。

文章的摘要如下：

*在这项工作中，我们提出了一种概念上简单而有效的方法来训练一个强大的双语多模态表示模型。从 OpenAI 发布的预训练多模态表示模型 CLIP 开始，我们将其文本编码器切换为预训练的多语言文本编码器 XLM-R，并通过教师学习和对比学习的两阶段训练模式对两种语言和图像表示进行对齐。我们通过对各种任务进行评估来验证我们的方法。我们在一系列任务上刷新了最新的性能记录，包括 ImageNet-CN、Flicker30k-CN 和 COCO-CN。此外，我们在几乎所有任务上都获得了与 CLIP 非常接近的性能，这表明可以简单地改变 CLIP 中的文本编码器以获得诸如多语言理解等扩展功能。

## 用法

AltCLIP 的用法与 CLIP 非常相似。与 CLIP 的区别在于文本编码器。请注意，我们使用双向注意力而不是单向注意力并且我们使用 XLM-R 的 [CLS] 标记来表示文本嵌入。
AltCLIP 是一个多模态的视觉和语言模型。它可用于图像-文本相似度和零样本图像分类。AltCLIP 使用类似于 ViT 的变压器来获取视觉特征，并使用双向语言模型获取文本特征。然后，将文本和视觉特征投影到具有相同维度的潜在空间中。然后使用投影图像和文本特征之间的点积作为相似度得分。

为了将图像馈送到 Transformer 编码器，将每个图像分割成固定大小且不重叠的序列补丁，然后进行线性嵌入。添加一个 [CLS] 标记作为整个图像的表示。作者还添加了绝对位置嵌入，并将结果向量的序列馈送到标准 Transformer 编码器。[`CLIPImageProcessor`] 可用于调整（或重新缩放）和规范化模型的图像。

[`AltCLIPProcessor`] 将 [`CLIPImageProcessor`] 和 [`XLMRobertaTokenizer`] 封装为一个实例，既可以对文本进行编码，也可以准备图像。下面的示例显示了如何使用 [`AltCLIPProcessor`] 和 [`AltCLIPModel`] 获取图像-文本相似度分数。

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

Tips:

此模型基于 `CLIPModel` 构建，因此可以像原始 CLIP 一样使用。
此模型由 [jongjyh](https://huggingface.co/jongjyh) 贡献。
## AltCLIPConfig
[[autodoc]] AltCLIPConfig
    - from_text_vision_configs
## AltCLIPTextConfig
[[autodoc]] AltCLIPTextConfig
## AltCLIPVisionConfig
[[autodoc]] AltCLIPVisionConfig
## AltCLIPProcessor
[[autodoc]] AltCLIPProcessor
## AltCLIPModel
[[autodoc]] AltCLIPModel
    - forward
    - get_text_features
    - get_image_features

## AltCLIPTextModel

[[autodoc]] AltCLIPTextModel
    - forward

## AltCLIPVisionModel

[[autodoc]] AltCLIPVisionModel
    - forward
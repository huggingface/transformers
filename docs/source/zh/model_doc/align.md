<!--版权所有 2023 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不附带任何形式的保证或条件。请参阅许可证以获取特定语言的权限和限制。⚠️请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法
在您的 Markdown 查看器中正确显示。渲染。
-->
# ALIGN

## 概述

ALIGN 模型是由 Chao Jia、Yinfei Yang、Ye Xia、Yi-Ting Chen、Zarana Parekh、Hieu Pham、Quoc V. Le、Yunhsuan Sung、Zhen Li、Tom Duerig 在 [《Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision》](https://arxiv.org/abs/2102.05918) 中提出的。ALIGN 是一个多模态的视觉和语言模型。它可用于图像文本相似度和零样本图像分类。ALIGN 采用双编码器架构，以 [EfficientNet](efficientnet) 作为其视觉编码器，以 [BERT](bert) 作为其文本编码器，并通过对比学习来对齐视觉和文本表示。与以往的工作不同，ALIGN 利用了一个庞大的嘈杂数据集，并表明语料库的规模可以通过简单的方法来实现 SOTA 的表示。

论文摘要如下所示：

*预训练表示对许多 NLP 和感知任务变得至关重要。虽然 NLP 中的表示学习已经过渡到对原始文本的训练，而不需要人工注释，但视觉和视觉语言表示仍然严重依赖于昂贵或需要专业知识的策划训练数据集。对于视觉应用程序，表示主要是使用具有显式类标签（如 ImageNet 或 OpenImages）的数据集进行学习。对于视觉语言，像 Conceptual Captions、MSCOCO 或 CLIP 这样的流行数据集都涉及非平凡的数据收集（和清理）过程。这个昂贵的策划过程限制了数据集的规模，从而阻碍了训练模型的规模化。在本文中，我们利用了一个超过十亿个图像替代文本对的嘈杂数据集，该数据集在 Conceptual Captions 数据集中不需要昂贵的过滤或后处理步骤。一个简单的双编码器架构学习了图像和文本对的视觉和语言表示之间的对齐关系，使用对比损失。我们展示了我们的语料库规模可以弥补其噪声，并在这样一个简单的学习方案下实现最先进的表示。我们的视觉表示在转移到 ImageNet 和 VTAB 等分类任务时表现出强大的性能。对齐的视觉和语言表示使得零样本图像分类成为可能，并在 Flickr30K 和 MSCOCO 图像-文本检索基准测试中取得了新的最先进的结果，即使与更复杂的交叉注意力模型相比也是如此。这些表示还可以实现具有复杂文本和文本+图像查询的跨模态搜索。*

## 用法

ALIGN 使用 EfficientNet 获取视觉特征，并使用 BERT 获取文本特征。然后将文本和视觉特征投影到具有相同维度的潜空间中。然后使用投影图像和文本特征之间的点积作为相似度分数。

[`AlignProcessor`] 将 [`EfficientNetImageProcessor`] 和 [`BertTokenizer`] 封装成一个实例，既可以编码文本，又可以预处理图像。下面的示例展示了如何使用 [`AlignProcessor`] 和 [`AlignModel`] 获取图像文本的相似度分数。
```python
import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["an image of a cat", "an image of a dog"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)
print(probs)
```

此模型由 [Alara Dirik](https://huggingface.co/adirik) 贡献。原始代码尚未发布，此实现基于 Kakao Brain 实现的原始论文。

## 资源

以下是官方的 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 ALIGN。

- 有关 ALIGN 和 COYO-700M 数据集的博客文章。- 一个零样本图像分类演示。- [`kakaobrain/align-base`](https://huggingface.co/kakaobrain/align-base) 模型的模型卡片。
如果您有兴趣提交要包含在此处的资源，请随时提出拉取请求，我们将对其进行审查。资源最好能够展示一些新东西，而不是重复现有的资源。

## AlignConfig
[[autodoc]] AlignConfig
    - from_text_vision_configs
## AlignTextConfig
[[autodoc]] AlignTextConfig
## AlignVisionConfig
[[autodoc]] AlignVisionConfig
## AlignProcessor
[[autodoc]] AlignProcessor
## AlignModel

[[autodoc]] AlignModel
    - forward
    - get_text_features
    - get_image_features

## AlignTextModel

[[autodoc]] AlignTextModel
    - forward

## AlignVisionModel

[[autodoc]] AlignVisionModel
    - forward
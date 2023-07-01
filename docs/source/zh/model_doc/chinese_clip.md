<!--版权2022年HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（“许可证”）授权；您除非符合许可证，否则不得使用此文件。您可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的“按现状”基础，无论是明示还是暗示的任何保证或条件。请参阅许可证特定语言下的权限和限制。
⚠️请注意，此文件是Markdown格式的，但包含了我们文档生成器的特定语法（类似于MDX），可能无法正确地在您的Markdown查看器中呈现。
-->
# Chinese-CLIP

## 概述

《Chinese-CLIP：中文对比视觉-语言预训练》一文由An Yang、Junshu Pan、Junyang Lin、Rui Men、Yichang Zhang、Jingren Zhou、Chang Zhou提出（[原文链接](https://arxiv.org/abs/2211.01335)）。Chinese-CLIP是CLIP（Radford等，2021年）在大规模中文图文对数据集上的实现。它能够进行跨模态检索，同时也可以作为视觉任务（如零样本图像分类、开放域目标检测等）的视觉骨干。原始的Chinese-CLIP代码可以在[此链接](https://github.com/OFA-Sys/Chinese-CLIP)找到。

来自论文的摘要如下：

*CLIP的巨大成功（Radford等，2021年）推动了视觉-语言预训练中对比学习的研究和应用。在这项工作中，我们构建了一个大规模的中文图文对数据集，其中大多数数据来自公开可用的数据集，然后我们在新数据集上预训练了中文CLIP模型。我们开发了5个中文CLIP模型，参数规模从7700万到9.58亿不等。此外，我们提出了一种两阶段的预训练方法，首先冻结图像编码器进行训练，然后对所有参数进行优化，以实现模型性能的增强。我们广泛的实验表明，中文CLIP在MUGE、Flickr30K-CN和COCO-CN的零样本学习和微调设置中都能达到最先进的性能，并且在ELEVATER基准测试（Li等，2022年）的零样本图像分类评估中能够取得竞争性的性能。我们已经发布了代码、预训练模型和演示。*

## 使用方法
下面的代码段展示了如何计算图像和文本特征以及相似度：
```python
>>> from PIL import Image
>>> import requests
>>> from transformers import ChineseCLIPProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> # Squirtle, Bulbasaur, Charmander, Pikachu in English
>>> texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

>>> # compute image feature
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
>>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

>>> # compute text features
>>> inputs = processor(text=texts, padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
>>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

>>> # compute image-text similarity scores
>>> inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
```

当前，我们在HF Model Hub上发布了以下规模的预训练Chinese-CLIP模型：
- [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
- [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
- [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)

The Chinese-CLIP model was contributed by [OFA-Sys](https://huggingface.co/OFA-Sys). 

## ChineseCLIPConfig

[[autodoc]] ChineseCLIPConfig
    - from_text_vision_configs

## ChineseCLIPTextConfig

[[autodoc]] ChineseCLIPTextConfig

## ChineseCLIPVisionConfig

[[autodoc]] ChineseCLIPVisionConfig

## ChineseCLIPImageProcessor

[[autodoc]] ChineseCLIPImageProcessor
    - preprocess

## ChineseCLIPFeatureExtractor

[[autodoc]] ChineseCLIPFeatureExtractor

## ChineseCLIPProcessor

[[autodoc]] ChineseCLIPProcessor

## ChineseCLIPModel

[[autodoc]] ChineseCLIPModel
    - forward
    - get_text_features
    - get_image_features

## ChineseCLIPTextModel

[[autodoc]] ChineseCLIPTextModel
    - forward

## ChineseCLIPVisionModel

[[autodoc]] ChineseCLIPVisionModel
    - forward
<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Chinese-CLIP

## Overview

The Chinese-CLIP model was proposed in [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
Chinese-CLIP is an implementation of CLIP (Radford et al., 2021) on a large-scale dataset of Chinese image-text pairs. It is capable of performing cross-modal retrieval and also playing as a vision backbone for vision tasks like zero-shot image classification, open-domain object detection, etc. The original Chinese-CLIP code is released [at this link](https://github.com/OFA-Sys/Chinese-CLIP).

The abstract from the paper is the following:

*The tremendous success of CLIP (Radford et al., 2021) has promoted the research and application of contrastive learning for vision-language pretraining. In this work, we construct a large-scale dataset of image-text pairs in Chinese, where most data are retrieved from publicly available datasets, and we pretrain Chinese CLIP models on the new dataset. We develop 5 Chinese CLIP models of multiple sizes, spanning from 77 to 958 million parameters. Furthermore, we propose a two-stage pretraining method, where the model is first trained with the image encoder frozen and then trained with all parameters being optimized, to achieve enhanced model performance. Our comprehensive experiments demonstrate that Chinese CLIP can achieve the state-of-the-art performance on MUGE, Flickr30K-CN, and COCO-CN in the setups of zero-shot learning and finetuning, and it is able to achieve competitive performance in zero-shot image classification based on the evaluation on the ELEVATER benchmark (Li et al., 2022). Our codes, pretrained models, and demos have been released.*

## Usage

The code snippet below shows how to compute image & text features and similarities:

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

Currently, we release the following scales of pretrained Chinese-CLIP models at HF Model Hub:

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
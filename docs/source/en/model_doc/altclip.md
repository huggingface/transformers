<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# AltCLIP

## Overview

The AltCLIP model was proposed in [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2) by Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, Ledell Wu. AltCLIP
(Altering the Language Encoder in CLIP) is a neural network trained on a variety of image-text and text-text pairs. By switching CLIP's
text encoder with a pretrained multilingual text encoder XLM-R, we could obtain very close performances with CLIP on almost all tasks, and extended original CLIP's capabilities such as multilingual understanding.

The abstract from the paper is the following:

*In this work, we present a conceptually simple and effective method to train a strong bilingual multimodal representation model. 
Starting from the pretrained multimodal representation model CLIP released by OpenAI, we switched its text encoder with a pretrained 
multilingual text encoder XLM-R, and aligned both languages and image representations by a two-stage training schema consisting of 
teacher learning and contrastive learning. We validate our method through evaluations of a wide range of tasks. We set new state-of-the-art 
performances on a bunch of tasks including ImageNet-CN, Flicker30k- CN, and COCO-CN. Further, we obtain very close performances with 
CLIP on almost all tasks, suggesting that one can simply alter the text encoder in CLIP for extended capabilities such as multilingual understanding.*

## Usage

The usage of AltCLIP is very similar to the CLIP. the difference between CLIP is the text encoder. Note that we use bidirectional attention instead of casual attention
and we take the [CLS] token in XLM-R to represent text embedding.

AltCLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image
classification. AltCLIP uses a ViT like transformer to get visual features and a bidirectional language model to get the text
features. Both the text and visual features are then projected to a latent space with identical dimension. The dot
product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image. The authors
also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
The [`CLIPImageProcessor`] can be used to resize (or rescale) and normalize images for the model.

The [`AltCLIPProcessor`] wraps a [`CLIPImageProcessor`] and a [`XLMRobertaTokenizer`] into a single instance to both
encode the text and prepare the images. The following example shows how to get the image-text similarity scores using
[`AltCLIPProcessor`] and [`AltCLIPModel`].


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

This model is build on `CLIPModel`, so use it like a original CLIP. 

This model was contributed by [jongjyh](https://huggingface.co/jongjyh).

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
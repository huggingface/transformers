<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLIP

## Overview

The CLIP model was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP
(Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be
instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing
for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

The abstract from the paper is the following:

*State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This
restricted form of supervision limits their generality and usability since additional labeled data is needed to specify
any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a
much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes
with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400
million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference
learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study
the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks
such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The
model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need
for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot
without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained
model weights at this https URL.*

## Usage

CLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image
classification. CLIP uses a ViT like transformer to get visual features and a causal language model to get the text
features. Both the text and visual features are then projected to a latent space with identical dimension. The dot
product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image. The authors
also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
The [`CLIPImageProcessor`] can be used to resize (or rescale) and normalize images for the model.

The [`CLIPTokenizer`] is used to encode the text. The [`CLIPProcessor`] wraps
[`CLIPImageProcessor`] and [`CLIPTokenizer`] into a single instance to both
encode the text and prepare the images. The following example shows how to get the image-text similarity scores using
[`CLIPProcessor`] and [`CLIPModel`].


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

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/openai/CLIP).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIP.

- A blog post on [How to fine-tune CLIP on 10,000 image-text pairs](https://huggingface.co/blog/fine-tune-clip-rsicd).
- CLIP is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it.
The resource should ideally demonstrate something new instead of duplicating an existing resource.

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

## FlaxCLIPTextModelWithProjection

[[autodoc]] FlaxCLIPTextModelWithProjection
    - __call__

## FlaxCLIPVisionModel

[[autodoc]] FlaxCLIPVisionModel
    - __call__

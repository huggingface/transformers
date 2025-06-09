<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MViTv2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The MViTv2 model was proposed in [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526) by Facebook Research. This work has been an inspiration for many further architectures and has been used for tasks such as image/video classification and detection.

The abstract from the paper:

In this paper, we study Multiscale Vision Transformers (MViTv2) as a unified architecture for image and video classification, as well as object detection. We present an improved version of MViT that incorporates decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where it outperforms prior work. We further compare MViTv2s' pooling attention to window attention mechanisms where it outperforms the latter in accuracy/compute. Without bells-and-whistles, MViTv2 has state-of-the-art performance in 3 domains: 88.8% accuracy on ImageNet classification, 58.7 boxAP on COCO object detection as well as 86.1% on Kinetics-400 video classification.

This model was contributed by [KamilaMila](https://huggingface.co/KamilaMila).
The original code can be found [here](https://github.com/facebookresearch/mvit). The Hugging Face implementation is also inspired by the implementation from Timm.

## Usage tips

There are many pretrained variants. Select your pretrained model based on your needs and resources.

## Resources

- [Image classification task guide](../tasks/image_classification)
- [Image classification task guide](../tasks/image_feature_extraction)

## MViTV2Config

[[autodoc]] MViTV2Config

## MViTV2Model

[[autodoc]] MViTV2Model
    - forward

## MViTV2ForVideoClassification

[[autodoc]] MViTV2ForVideoClassification
    - forward

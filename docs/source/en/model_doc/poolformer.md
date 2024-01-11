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

# PoolFormer

## Overview

The PoolFormer model was proposed in [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)  by Sea AI Labs. Instead of designing complicated token mixer to achieve SOTA performance, the target of this work is to demonstrate the competence of transformer models largely stem from the general architecture MetaFormer.

The abstract from the paper is the following:

*Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only the most basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design.*

The figure below illustrates the architecture of PoolFormer. Taken from the [original paper](https://arxiv.org/abs/2111.11418).

<img width="600" src="https://user-images.githubusercontent.com/15921929/142746124-1ab7635d-2536-4a0e-ad43-b4fe2c5a525d.png"/>

This model was contributed by [heytanay](https://huggingface.co/heytanay). The original code can be found [here](https://github.com/sail-sg/poolformer).

## Usage tips

- PoolFormer has a hierarchical architecture, where instead of Attention, a simple Average Pooling layer is present. All checkpoints of the model can be found on the [hub](https://huggingface.co/models?other=poolformer).
- One can use [`PoolFormerImageProcessor`] to prepare images for the model.
- As most models, PoolFormer comes in different sizes, the details of which can be found in the table below.

| **Model variant** | **Depths**    | **Hidden sizes**    | **Params (M)** | **ImageNet-1k Top 1** |
| :---------------: | ------------- | ------------------- | :------------: | :-------------------: |
| s12               | [2, 2, 6, 2]  | [64, 128, 320, 512] | 12             | 77.2                  |
| s24               | [4, 4, 12, 4] | [64, 128, 320, 512] | 21             | 80.3                  |
| s36               | [6, 6, 18, 6] | [64, 128, 320, 512] | 31             | 81.4                  |
| m36               | [6, 6, 18, 6] | [96, 192, 384, 768] | 56             | 82.1                  |
| m48               | [8, 8, 24, 8] | [96, 192, 384, 768] | 73             | 82.5                  |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with PoolFormer.

<PipelineTag pipeline="image-classification"/>

- [`PoolFormerForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## PoolFormerConfig

[[autodoc]] PoolFormerConfig

## PoolFormerFeatureExtractor

[[autodoc]] PoolFormerFeatureExtractor
    - __call__

## PoolFormerImageProcessor

[[autodoc]] PoolFormerImageProcessor
    - preprocess

## PoolFormerModel

[[autodoc]] PoolFormerModel
    - forward

## PoolFormerForImageClassification

[[autodoc]] PoolFormerForImageClassification
    - forward

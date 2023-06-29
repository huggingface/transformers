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

# ViLT

## Overview

The ViLT model was proposed in [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)
by Wonjae Kim, Bokyung Son, Ildoo Kim. ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design
for Vision-and-Language Pre-training (VLP).

The abstract from the paper is the following:

*Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks.
Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision
(e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we
find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more
computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive
power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model,
Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically
simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of
times faster than previous VLP models, yet with competitive or better downstream task performance.*

Tips:

- The quickest way to get started with ViLT is by checking the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT)
  (which showcase both inference and fine-tuning on custom data).
- ViLT is a model that takes both `pixel_values` and `input_ids` as input. One can use [`ViltProcessor`] to prepare data for the model.
  This processor wraps a image processor (for the image modality) and a tokenizer (for the language modality) into one.
- ViLT is trained with images of various sizes: the authors resize the shorter edge of input images to 384 and limit the longer edge to
  under 640 while preserving the aspect ratio. To make batching of images possible, the authors use a `pixel_mask` that indicates
  which pixel values are real and which are padding. [`ViltProcessor`] automatically creates this for you.
- The design of ViLT is very similar to that of a standard Vision Transformer (ViT). The only difference is that the model includes
  additional embedding layers for the language modality.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg"
alt="drawing" width="600"/>

<small> ViLT architecture. Taken from the <a href="https://arxiv.org/abs/2102.03334">original paper</a>. </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/dandelin/ViLT).


Tips:

- The PyTorch version of this model is only available in torch 1.10 and higher.

## ViltConfig

[[autodoc]] ViltConfig

## ViltFeatureExtractor

[[autodoc]] ViltFeatureExtractor
    - __call__

## ViltImageProcessor

[[autodoc]] ViltImageProcessor
    - preprocess

## ViltProcessor

[[autodoc]] ViltProcessor
    - __call__

## ViltModel

[[autodoc]] ViltModel
    - forward

## ViltForMaskedLM

[[autodoc]] ViltForMaskedLM
    - forward

## ViltForQuestionAnswering

[[autodoc]] ViltForQuestionAnswering
    - forward

## ViltForImagesAndTextClassification

[[autodoc]] ViltForImagesAndTextClassification
    - forward

## ViltForImageAndTextRetrieval

[[autodoc]] ViltForImageAndTextRetrieval
    - forward

## ViltForTokenClassification

[[autodoc]] ViltForTokenClassification
    - forward

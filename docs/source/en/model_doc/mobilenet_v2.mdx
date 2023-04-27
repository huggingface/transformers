<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MobileNet V2

## Overview

The MobileNet model was proposed in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.

The abstract from the paper is the following:

*In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3.*

*The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters.*

Tips:

- The checkpoints are named **mobilenet\_v2\_*depth*\_*size***, for example **mobilenet\_v2\_1.0\_224**, where **1.0** is the depth multiplier (sometimes also referred to as "alpha" or the width multiplier) and **224** is the resolution of the input images the model was trained on.

- Even though the checkpoint is trained on images of specific size, the model will work on images of any size. The smallest supported image size is 32x32.

- One can use [`MobileNetV2ImageProcessor`] to prepare images for the model.

- The available image classification checkpoints are pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes). However, the model predicts 1001 classes: the 1000 classes from ImageNet plus an extra “background” class (index 0).

- The segmentation model uses a [DeepLabV3+](https://arxiv.org/abs/1802.02611) head. The available semantic segmentation checkpoints are pre-trained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

- The original TensorFlow checkpoints use different padding rules than PyTorch, requiring the model to determine the padding amount at inference time, since this depends on the input image size. To use native PyTorch padding behavior, create a [`MobileNetV2Config`] with `tf_padding = False`.

Unsupported features:

- The [`MobileNetV2Model`] outputs a globally pooled version of the last hidden state. In the original model it is possible to use an average pooling layer with a fixed 7x7 window and stride 1 instead of global pooling. For inputs that are larger than the recommended image size, this gives a pooled output that is larger than 1x1. The Hugging Face implementation does not support this.

- The original TensorFlow checkpoints include quantized models. We do not support these models as they include additional "FakeQuantization" operations to unquantize the weights.

- It's common to extract the output from the expansion layers at indices 10 and 13, as well as the output from the final 1x1 convolution layer, for downstream purposes. Using `output_hidden_states=True` returns the output from all intermediate layers. There is currently no way to limit this to specific layers.

- The DeepLabV3+ segmentation head does not use the final convolution layer from the backbone, but this layer gets computed anyway. There is currently no way to tell [`MobileNetV2Model`] up to which layer it should run.

This model was contributed by [matthijs](https://huggingface.co/Matthijs). The original code and weights can be found [here for the main model](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) and [here for DeepLabV3+](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MobileNetV2.

<PipelineTag pipeline="image-classification"/>

- [`MobileNetV2ForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

**Semantic segmentation**
- [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## MobileNetV2Config

[[autodoc]] MobileNetV2Config

## MobileNetV2FeatureExtractor

[[autodoc]] MobileNetV2FeatureExtractor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2ImageProcessor

[[autodoc]] MobileNetV2ImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2Model

[[autodoc]] MobileNetV2Model
    - forward

## MobileNetV2ForImageClassification

[[autodoc]] MobileNetV2ForImageClassification
    - forward

## MobileNetV2ForSemanticSegmentation

[[autodoc]] MobileNetV2ForSemanticSegmentation
    - forward

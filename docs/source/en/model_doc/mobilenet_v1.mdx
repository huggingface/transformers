<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MobileNet V1

## Overview

The MobileNet model was proposed in [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.

The abstract from the paper is the following:

*We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.*

Tips:

- The checkpoints are named **mobilenet\_v1\_*depth*\_*size***, for example **mobilenet\_v1\_1.0\_224**, where **1.0** is the depth multiplier (sometimes also referred to as "alpha" or the width multiplier) and **224** is the resolution of the input images the model was trained on.

- Even though the checkpoint is trained on images of specific size, the model will work on images of any size. The smallest supported image size is 32x32.

- One can use [`MobileNetV1ImageProcessor`] to prepare images for the model.

- The available image classification checkpoints are pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes). However, the model predicts 1001 classes: the 1000 classes from ImageNet plus an extra “background” class (index 0).

- The original TensorFlow checkpoints use different padding rules than PyTorch, requiring the model to determine the padding amount at inference time, since this depends on the input image size. To use native PyTorch padding behavior, create a [`MobileNetV1Config`] with `tf_padding = False`.

Unsupported features:

- The [`MobileNetV1Model`] outputs a globally pooled version of the last hidden state. In the original model it is possible to use a 7x7 average pooling layer with stride 2 instead of global pooling. For larger inputs, this gives a pooled output that is larger than 1x1 pixel. The HuggingFace implementation does not support this.

- It is currently not possible to specify an `output_stride`. For smaller output strides, the original model invokes dilated convolution to prevent the spatial resolution from being reduced further. The output stride of the HuggingFace model is always 32.

- The original TensorFlow checkpoints include quantized models. We do not support these models as they include additional "FakeQuantization" operations to unquantize the weights.

- It's common to extract the output from the pointwise layers at indices 5, 11, 12, 13 for downstream purposes. Using `output_hidden_states=True` returns the output from all intermediate layers. There is currently no way to limit this to specific layers.

This model was contributed by [matthijs](https://huggingface.co/Matthijs). The original code and weights can be found [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MobileNetV1.

<PipelineTag pipeline="image-classification"/>

- [`MobileNetV1ForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## MobileNetV1Config

[[autodoc]] MobileNetV1Config

## MobileNetV1FeatureExtractor

[[autodoc]] MobileNetV1FeatureExtractor
    - preprocess

## MobileNetV1ImageProcessor

[[autodoc]] MobileNetV1ImageProcessor
    - preprocess

## MobileNetV1Model

[[autodoc]] MobileNetV1Model
    - forward

## MobileNetV1ForImageClassification

[[autodoc]] MobileNetV1ForImageClassification
    - forward

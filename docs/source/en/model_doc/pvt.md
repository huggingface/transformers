<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pyramid Vision Transformer (PVT)

## Overview

The PVT model was proposed in
[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)
by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao. The PVT is a type of
vision transformer that utilizes a pyramid structure to make it an effective backbone for dense prediction tasks. Specifically
it allows for more fine-grained inputs (4 x 4 pixels per patch) to be used, while simultaneously shrinking the sequence length
of the Transformer as it deepens - reducing the computational cost. Additionally, a spatial-reduction attention (SRA) layer
is used to further reduce the resource consumption when learning high-resolution features.

The abstract from the paper is the following:

*Although convolutional neural networks (CNNs) have achieved great success in computer vision, this work investigates a 
simpler, convolution-free backbone network useful for many dense prediction tasks. Unlike the recently proposed Vision 
Transformer (ViT) that was designed for image classification specifically, we introduce the Pyramid Vision Transformer 
(PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several 
merits compared to current state of the arts. Different from ViT that typically yields low resolution outputs and 
incurs high computational and memory costs, PVT not only can be trained on dense partitions of an image to achieve high 
output resolution, which is important for dense prediction, but also uses a progressive shrinking pyramid to reduce the 
computations of large feature maps. PVT inherits the advantages of both CNN and Transformer, making it a unified 
backbone for various vision tasks without convolutions, where it can be used as a direct replacement for CNN backbones. 
We validate PVT through extensive experiments, showing that it boosts the performance of many downstream tasks, including
object detection, instance and semantic segmentation. For example, with a comparable number of parameters, PVT+RetinaNet 
achieves 40.4 AP on the COCO dataset, surpassing ResNet50+RetinNet (36.3 AP) by 4.1 absolute AP (see Figure 2). We hope 
that PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future research.*

This model was contributed by [Xrenya](<https://huggingface.co/Xrenya). The original code can be found [here](https://github.com/whai362/PVT).


- PVTv1 on ImageNet-1K

| **Model variant**  |**Size** |**Acc@1**|**Params (M)**|
|--------------------|:-------:|:-------:|:------------:|
| PVT-Tiny           |    224  |   75.1  |     13.2     |
| PVT-Small          |    224  |   79.8  |     24.5     |
| PVT-Medium         |    224  |   81.2  |     44.2     |
| PVT-Large          |    224  |   81.7  |     61.4     |


## PvtConfig

[[autodoc]] PvtConfig

## PvtImageProcessor

[[autodoc]] PvtImageProcessor
    - preprocess

## PvtForImageClassification

[[autodoc]] PvtForImageClassification
    - forward

## PvtModel

[[autodoc]] PvtModel
    - forward

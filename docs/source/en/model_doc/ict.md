<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ICT

## Overview

The ICT model was proposed in [High-Fidelity Pluralistic Image Completion with Transformers](https://arxiv.org/abs/2103.14031) 
by Ziyu Wan, Jingbo Zhang, Dongdong Chen, Jing Liao. ICT (Image Completion with Transformers) leverages both a 
transformer and CNNs by decoupling image completion into two steps: pluralistic appearance priors reconstruction with a 
transformer to recover the coherent image structures, and low-resolution upsampling with CNNs to replenish ﬁne textures.

The abstract from the paper is the following:

*Image completion has made tremendous progress with convolutional neural networks (CNNs), because of their powerful texture modeling capacity. However, due to some inherent properties (e.g., local inductive prior, spatial-invariant kernels), CNNs do not perform well in understanding global structures or naturally support pluralistic completion. Recently, transformers demonstrate their power in modeling the long-term relationship and generating diverse results, but their computation complexity is quadratic to input length, thus hampering the application in processing high-resolution images. This paper brings the best of both worlds to pluralistic image completion: appearance prior reconstruction with transformer and texture replenishment with CNN. The former transformer recovers pluralistic coherent structures together with some coarse textures, while the latter CNN enhances the local texture details of coarse priors guided by the high-resolution masked images. The proposed method vastly outperforms state-of-the-art methods in terms of three aspects: 1) large performance boost on image fidelity even compared to deterministic completion methods; 2) better diversity and higher fidelity for pluralistic completion; 3) exceptional generalization ability on large masks and generic dataset, like ImageNet.*

Tips:

- Unlike auto-regressive methods, in order to make the transformer model capable of completing the missing regions by 
  considering all the available context, this model optimizes the log-likelihood objective of missing pixels 
  bi-directionally conditions, which is inspired by the masked language model like BERT.
- The computational cost of multi-head attention increases quadratically, so the appearance priors is resized to 
  low-resolution versions, which contains structural information and coarse textures only. But the dimension is further 
  reduced by using an extra visual vocabulary (512 × 3) which is generated using k-means cluster centers of the whole 
  ImageNet RGB pixel spaces.
- Three available checkpoints are trained on [ImageNet](https://www.image-net.org/challenges/LSVRC), 
  [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [Places2](http://places2.csail.mit.edu/).

This model was contributed by [Sheon Han](https://huggingface.co/sheonhan).
The original code can be found [here](https://github.com/raywzy/ICT).


## IctConfig

[[autodoc]] IctConfig

## IctImageProcessor

[[autodoc]] IctImageProcessor
    - preprocess

## IctModel

[[autodoc]] IctModel
    - forward

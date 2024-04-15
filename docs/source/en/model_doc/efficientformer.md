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

# EfficientFormer

## Overview

The EfficientFormer model was proposed in [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191) 
by Yanyu Li, Geng Yuan, Yang Wen, Eric Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.  EfficientFormer proposes a
dimension-consistent pure transformer that can be run on mobile devices for dense prediction tasks like image classification, object
detection and semantic segmentation.

The abstract from the paper is the following:

*Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. 
However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally 
times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly 
challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation 
complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still 
unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? 
To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. 
Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm. 
Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. 
Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. 
Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on 
iPhone 12 (compiled with CoreML), which { runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1),} and our largest model, 
EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can 
reach extremely low latency on mobile devices while maintaining high performance.*

This model was contributed by [novice03](https://huggingface.co/novice03) and [Bearnardd](https://huggingface.co/Bearnardd).
The original code can be found [here](https://github.com/snap-research/EfficientFormer). The TensorFlow version of this model was added by [D-Roberts](https://huggingface.co/D-Roberts).

## Documentation resources

- [Image classification task guide](../tasks/image_classification)

## EfficientFormerConfig

[[autodoc]] EfficientFormerConfig

## EfficientFormerImageProcessor

[[autodoc]] EfficientFormerImageProcessor
    - preprocess

<frameworkcontent>
<pt>

## EfficientFormerModel

[[autodoc]] EfficientFormerModel
    - forward

## EfficientFormerForImageClassification

[[autodoc]] EfficientFormerForImageClassification
    - forward

## EfficientFormerForImageClassificationWithTeacher

[[autodoc]] EfficientFormerForImageClassificationWithTeacher
    - forward

</pt>
<tf>

## TFEfficientFormerModel

[[autodoc]] TFEfficientFormerModel
    - call

## TFEfficientFormerForImageClassification

[[autodoc]] TFEfficientFormerForImageClassification
    - call

## TFEfficientFormerForImageClassificationWithTeacher

[[autodoc]] TFEfficientFormerForImageClassificationWithTeacher
    - call

</tf>
</frameworkcontent>
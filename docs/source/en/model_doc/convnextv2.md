<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-01-02 and added to Hugging Face Transformers on 2023-03-14.*

# ConvNeXt V2

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

## Overview

The ConvNeXt V2 model was proposed in [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://huggingface.co/papers/2301.00808) by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
ConvNeXt V2 is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers, and a successor of [ConvNeXT](convnext).

The abstract from the paper is the following:

*Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt, have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked  autoencoders (MAE). However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M-parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves a state-of-the-art 88.9% accuracy using only public training data.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png"
alt="drawing" width="600"/>

<small> ConvNeXt V2 architecture. Taken from the <a href="https://huggingface.co/papers/2301.00808">original paper</a>.</small>

This model was contributed by [adirik](https://huggingface.co/adirik). The original code can be found [here](https://github.com/facebookresearch/ConvNeXt-V2).

> [!TIP]
> This model was contributed by [adirik](https://huggingface.co/adirik).
>
> Click on the **ConvNeXt V2** models in the right sidebar for more examples of how to apply ConvNeXt V2 to different **image-classification** tasks.

## Intended uses & limitations

**Use for**
- Image classification out of the box (ImageNet-1k/22k fine-tuned checkpoints).
- As a **backbone** to extract multi-scale feature maps for detection/segmentation tasks.

**Limitations / caveats**
- Most layers are `Conv2d`. Quantization methods that only target linear layers (e.g. 8/4-bit with bitsandbytes) will primarily affect the classification head and yield modest memory savings compared to transformer LLMs.
- Accuracy is sensitive to input resolution and preprocessing. Match your evaluation transforms to the checkpoint’s training recipe (e.g., 224 vs 384).

## How to use (quickstart)

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline
from PIL import Image

clf = pipeline("image-classification", model="facebook/convnextv2-tiny-1k-224")
img = Image.open("cat.jpg")
print(clf(img)[:3])  # top-3 predictions

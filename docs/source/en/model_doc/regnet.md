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

# RegNet

## Overview

The RegNet model was proposed in [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) by Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár.

The authors design search spaces to perform Neural Architecture Search (NAS). They first start from a high dimensional search space and iteratively reduce the search space by empirically applying constraints based on the best-performing models sampled by the current search space.

The abstract from the paper is the following:

*In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.*

Tips:

- One can use [`AutoImageProcessor`] to prepare images for the model.
- The huge 10B model from [Self-supervised Pretraining of Visual Features in the Wild](https://arxiv.org/abs/2103.01988), trained on one billion Instagram images, is available on the [hub](https://huggingface.co/facebook/regnet-y-10b-seer)

This model was contributed by [Francesco](https://huggingface.co/Francesco). The TensorFlow version of the model
was contributed by [sayakpaul](https://huggingface.com/sayakpaul) and [ariG23498](https://huggingface.com/ariG23498).
The original code can be found [here](https://github.com/facebookresearch/pycls).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RegNet.

<PipelineTag pipeline="image-classification"/>

- [`RegNetForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## RegNetConfig

[[autodoc]] RegNetConfig


## RegNetModel

[[autodoc]] RegNetModel
    - forward


## RegNetForImageClassification

[[autodoc]] RegNetForImageClassification
    - forward

## TFRegNetModel

[[autodoc]] TFRegNetModel
    - call


## TFRegNetForImageClassification

[[autodoc]] TFRegNetForImageClassification
    - call


## FlaxRegNetModel

[[autodoc]] FlaxRegNetModel
    - __call__


## FlaxRegNetForImageClassification

[[autodoc]] FlaxRegNetForImageClassification
    - __call__
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

# Convolutional Vision Transformer (CvT)

## Overview

The CCT model was proposed in [Escaping the Big Data Paradigm with
Compact Transformers](https://arxiv.org/abs/2104.05704) by Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, Humphrey Shi.

The abstract from the paper is the following:

*With the rise of Transformers as the standard for language processing, and their advancements in computer vi-
sion, there has been a corresponding growth in parameter
size and amounts of training data. Many have come to be-
lieve that because of this, transformers are not suitable for
small sets of data. This trend leads to concerns such as:
limited availability of data in certain scientific domains and
the exclusion of those with limited resource from research
in the field. In this paper, we aim to present an approach for
small-scale learning by introducing Compact Transformers.
We show for the first time that with the right size, convolu-
tional tokenization, transformers can avoid overfitting and
outperform state-of-the-art CNNs on small datasets. Our
models are flexible in terms of model size, and can have as
little as 0.28M parameters while achieving competitive re-
sults. Our best model can reach 98% accuracy when train-
ing from scratch on CIFAR-10 with only 3.7M parameters,
which is a significant improvement in data-efficiency over
previous Transformer based models being over 10x smaller
than other transformers and is 15% the size of ResNet50
while achieving similar performance. CCT also outper-
forms many modern CNN based approaches, and even some
recent NAS-based approaches. Additionally, we obtain a
new SOTA result on Flowers-102 with 99.76% top-1 accu-
racy, and improve upon the existing baseline on ImageNet
(82.71% accuracy with 29% as many parameters as ViT),
as well as NLP tasks. Our simple and compact design for
transformers makes them more feasible to study for those
with limited computing resources and/or dealing with small
datasets, while extending existing research efforts in data
efficient transformers.*


This model was contributed by [rishabbala](https://huggingface.co/rishabbala). The original code can be found [here](https://github.com/SHI-Labs/Compact-Transformers).

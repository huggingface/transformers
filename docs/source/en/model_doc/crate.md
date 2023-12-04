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

# CRATE

## Overview

The CRATE model was proposed in [White-Box Transformers via Sparse Rate Reduction: Compression Is All There Is?](https://arxiv.org/abs/2311.13110) by Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Hao Bai, Yuexiang Zhai, Benjamin D. Haeffele, Yi Ma.
This paper presents a novel perspective on representation learning, proposing that its natural goal should be to compress and transform data distributions, like sets of tokens, into low-dimensional Gaussian mixtures on incoherent subspaces. The authors introduce 'sparse rate reduction' as a measure to assess the quality of representations, aiming to maximize both intrinsic information gain and extrinsic sparsity. They suggest that popular deep network architectures, such as transformers, can be seen as optimizing this measure through iterative schemes. Specifically, the paper derives a transformer block from this perspective, where the multi-head self-attention operator compresses representations, and the multi-layer perceptron sparsifies them. This approach leads to a new class of transformer-like architectures called CRATE, which are mathematically interpretable. The paper also establishes a link between denoising and compression, showing that CRATE architectures can function as both encoders and decoders. Empirical results demonstrate that these networks can effectively compress and sparsify representations from large-scale image and text datasets, achieving performance comparable to advanced transformer-based models. This research suggests significant potential for this computational framework in narrowing the gap between deep learning theory and practice, centered around data compression. The code for this study is made publicly available.

The abstract from the paper is the following:

*In this paper, we contend that a natural objective of representation learning is to compress and transform the distribution of the data, say sets of tokens, towards a low-dimensional Gaussian mixture supported on incoherent subspaces. The goodness of such a representation can be evaluated by a principled measure, called sparse rate reduction, that simultaneously maximizes the intrinsic information gain and extrinsic sparsity of the learned representation. From this perspective, popular deep network architectures, including transformers, can be viewed as realizing iterative schemes to optimize this measure. Particularly, we derive a transformer block from alternating optimization on parts of this objective: the multi-head self-attention operator compresses the representation by implementing an approximate gradient descent step on the coding rate of the features, and the subsequent multi-layer perceptron sparsifies the features. This leads to a family of white-box transformer-like deep network architectures, named CRATE, which are mathematically fully interpretable. We show, by way of a novel connection between denoising and compression, that the inverse to the aforementioned compressive encoding can be realized by the same class of CRATE architectures. Thus, the so-derived white-box architectures are universal to both encoders and decoders. Experiments show that these networks, despite their simplicity, indeed learn to compress and sparsify representations of large-scale real-world image and text datasets, and achieve performance very close to highly engineered transformer-based models: ViT, MAE, DINO, BERT, and GPT2. We believe the proposed computational framework demonstrates great potential in bridging the gap between theory and practice of deep learning, from a unified perspective of data compression. Code is available at: https://github.com/Ma-Lab-Berkeley/CRATE.*

Tips:

This model was contributed by [JackBAI](https://huggingface.co/JackBAI). This model serves as the official release of the CRATE-BERT implementation for text. The CRATE-GPT2 implementation of text is currently released in a separate repository and will be migrated to HF soon.

## CrateConfig

[[autodoc]] CrateConfig

## CrateModel

[[autodoc]] CrateModel
    - forward

## CrateForCausalLM

[[autodoc]] CrateForCausalLM
    - forward

## CrateForMaskedLM

[[autodoc]] CrateForMaskedLM
    - forward

## CrateForSequenceClassification

[[autodoc]] CrateForSequenceClassification
    - forward

## CrateForMultipleChoice

[[autodoc]] CrateForMultipleChoice
    - forward

## CrateForTokenClassification

[[autodoc]] CrateForTokenClassification
    - forward

## CrateForQuestionAnswering

[[autodoc]] CrateForQuestionAnswering
    - forward

</pt>
<tf>

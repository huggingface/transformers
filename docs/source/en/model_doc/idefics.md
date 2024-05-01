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

# IDEFICS

## Overview

The IDEFICS model was proposed in [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents
](https://huggingface.co/papers/2306.16527
) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

The abstract from the paper is the following:

*Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks that require reasoning over one or multiple images to generate a text. However, the datasets used to train these models have not been released, and the collection process has not been fully specified. We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset's content. To show the viability of OBELISC, we train an 80 billion parameters vision and language model on the dataset and obtain competitive performance on various multimodal benchmarks. We release the code to reproduce the dataset along with the dataset itself.*

This model was contributed by [HuggingFaceM4](https://huggingface.co/HuggingFaceM4). The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>). (TODO: don't have a public link yet).


<Tip warning={true}>

IDEFICS modeling code in Transformers is for finetuning and inferencing the pre-trained IDEFICS models.

To train a new IDEFICS model from scratch use the m4 codebase (a link will be provided once it's made public)

</Tip>


## IdeficsConfig

[[autodoc]] IdeficsConfig

## IdeficsModel

[[autodoc]] IdeficsModel
    - forward

## IdeficsForVisionText2Text

[[autodoc]] IdeficsForVisionText2Text
    - forward

## IdeficsImageProcessor

[[autodoc]] IdeficsImageProcessor
    - preprocess

## IdeficsProcessor

[[autodoc]] IdeficsProcessor
    - __call__

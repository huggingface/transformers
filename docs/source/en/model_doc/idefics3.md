<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Idefics3

## Overview

The Idefics3 model was proposed in [Building and better understanding vision-language models: insights and future directions](https://huggingface.co/papers/2408.12637) by Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon.

Idefics3 is an adaptation of the Idefics2 model with three main differences:
- the use of Llama3 for the text model.
- an updated processing logic for the images.
- The removal of the perceiver.

The resolutions of input images can be directly controlled, and they are decomposed into
patches, or not, depending on the resolution. See [Idefics2] for more details on the model architecture.

The abstract from the paper is the following:

*The field of vision-language models (VLMs), which take images and texts as inputs and output texts, is rapidly evolving and has yet to reach consensus on several key aspects of the development pipeline, including data, architecture, and training methods. This paper can be seen as a tutorial for building a VLM. We begin by providing a comprehensive overview of the current state-of-the-art approaches, highlighting the strengths and weaknesses of each, addressing the major challenges in the field, and suggesting promising research directions for underexplored areas. We then walk through the practical steps to build Idefics3-8B, a powerful VLM that significantly outperforms its predecessor Idefics2-8B, while being trained efficiently, exclusively on open datasets, and using a straightforward pipeline. These steps include the creation of Docmatix, a dataset for improving document understanding capabilities, which is 240 times larger than previously available datasets. We release the model along with the datasets created for its training.*

Tips:

- The input given to the model will be resized by default such that the longest side is 4*364. For faster inference, set `do_resize` to `False`.

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [andimarafioti](https://huggingface.co/andito).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## Idefics3Config

[[autodoc]] Idefics3Config


## Idefics3Model

[[autodoc]] Idefics3Model
    - forward

## Idefics3ForConditionalGeneration

[[autodoc]] Idefics3ForConditionalGeneration
    - forward


## Idefics3ImageProcessor
[[autodoc]] Idefics3ImageProcessor
    - preprocess


## Idefics3Processor
[[autodoc]] Idefics3Processor
    - __call__

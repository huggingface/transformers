<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UDOP

## Overview

The UDOP model was proposed in [Unifying Vision, Text, and Layout for Universal Document Processing](https://arxiv.org/abs/2212.02623) by Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal.
UDOP adopts an encoder-decoder Transformer architecture (based on [T5](t5)) for document AI tasks like document imageclassification, document parsing and document visual question answering.

The abstract from the paper is the following:

We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 9 Document AI tasks, e.g., document understanding and QA, across diverse data domains like finance reports, academic papers, and websites. UDOP ranks first on the leaderboard of the Document Understanding Benchmark (DUE).*

Tips:

- At inference time, it's recommended to use the `generate` method to autoregressively generate text given a document image.
- UDOP relies on an OCR engine of choice. By default, [`UdopProcessor`] uses the Tesseract engine to extract a list of words and boxes (coordinates) from a given document.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/microsoft/UDOP).


## UdopConfig

[[autodoc]] UdopConfig

## UdopTokenizer

[[autodoc]] UdopTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## UdopTokenizerFast

[[autodoc]] UdopTokenizerFast

## UdopImageProcessor

[[autodoc]] UdopImageProcessor
    - preprocess

## UdopProcessor

[[autodoc]] UdopProcessor
    - __call__

## UdopModel

[[autodoc]] UdopModel
    - forward

## UdopForConditionalGeneration

[[autodoc]] UdopForConditionalGeneration
    - forward

## UdopEncoderModel

[[autodoc]] UdopEncoderModel
    - forward
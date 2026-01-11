<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VideoPrism

## Overview

The VideoPrism model was proposed in the paper [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://huggingface.co/papers/2402.13217) by Google DeepMind ([blog post](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)).

VideoPrism is a general-purpose video encoder that tackles diverse video understanding tasks with a single frozen model. The model is pretrained on a large-scale heterogeneous corpus containing 36M high-quality video-caption pairs and 582M video clips with noisy parallel text (e.g., ASR transcripts). The pretraining approach improves upon masked autoencoding through global-local distillation of semantic video embeddings and a token shuffling scheme, enabling the model to focus primarily on the video modality while leveraging text associated with videos. VideoPrism achieves state-of-the-art performance on 31 out of 33 video understanding benchmarks across four broad task groups, from web video question answering to computer vision for science.

Tips:

- VideoPrism uses a factorized spatio-temporal encoder architecture, processing videos through separate spatial and temporal transformers.
- The model supports video-text contrastive learning through `VideoPrismClipModel`, which combines a video encoder and a text encoder.
- For video classification tasks, use `VideoPrismForVideoClassification` which adds a classification head on top of the video encoder.
- The default input resolution is 288x288 pixels with 16 frames per video clip.
- The vision encoder can be used standalone via `VideoPrismVisionModel` for extracting video features.

This model was contributed by [MHRDYN7](https://github.com/MHRDYN7) and reviewed by [qubvel](https://github.com/qubvel) & [zucchini-nlp](https://github.com/zucchini-nlp).
The original code can be found [here](https://github.com/google-deepmind/videoprism).

## VideoPrismVisionConfig

[[autodoc]] VideoPrismVisionConfig

## VideoPrismTextConfig

[[autodoc]] VideoPrismTextConfig

## VideoPrismConfig

[[autodoc]] VideoPrismConfig

## VideoPrismVideoProcessor

[[autodoc]] VideoPrismVideoProcessor

## VideoPrismTokenizer

[[autodoc]] VideoPrismTokenizer

## VideoPrismProcessor

[[autodoc]] VideoPrismProcessor

## VideoPrismVisionModel

[[autodoc]] VideoPrismVisionModel
    - forward

## VideoPrismVideoModel

[[autodoc]] VideoPrismVideoModel
    - forward

## VideoPrismTextModel

[[autodoc]] VideoPrismTextModel
    - forward

## VideoPrismClipModel

[[autodoc]] VideoPrismClipModel
    - forward

## VideoPrismForVideoClassification

[[autodoc]] VideoPrismForVideoClassification
    - forward

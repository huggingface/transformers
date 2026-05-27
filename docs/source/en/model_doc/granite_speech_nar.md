<!--Copyright 2026 IBM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-03-09 and added to Hugging Face Transformers on 2026-05-26.*

# GraniteSpeechNar

## Overview

GraniteSpeechNar is a non-autoregressive (NAR) speech recognition model based on [NLE: Non-autoregressive LLM-based ASR by Transcript Editing](https://huggingface.co/papers/2603.08397). It formulates ASR as conditional transcript editing, achieving fully parallel prediction with significant speedups over autoregressive baselines.

The model consists of:

1. **Conformer Encoder**: A conformer encoder trained with CTC on BPE targets, using block-attention and self-conditioned CTC from the middle layer.

2. **QFormer Projector**: A windowed query-transformer that maps multi-layer encoder features to the LLM embedding space with temporal downsampling.

3. **Bidirectional Granite LLM**: A Granite language model with bidirectional (non-causal) attention that refines CTC predictions in a single forward pass.

The model performs inference in a single pass: the encoder produces initial CTC predictions, which are interleaved with blank insertion slots (exploiting the identity mapping bias of Transformers) and fed alongside projected audio embeddings to the bidirectional LLM for refinement via a latent alignment objective.

This model was contributed by [Avihu Dekel](https://huggingface.co/Avihu).

## GraniteSpeechNarConfig

[[autodoc]] GraniteSpeechNarConfig

## GraniteSpeechNarEncoderConfig

[[autodoc]] GraniteSpeechNarEncoderConfig

## GraniteSpeechNarProjectorConfig

[[autodoc]] GraniteSpeechNarProjectorConfig

## GraniteSpeechNarProcessor

[[autodoc]] GraniteSpeechNarProcessor
    - __call__
    - batch_decode

## GraniteSpeechNarFeatureExtractor

[[autodoc]] GraniteSpeechNarFeatureExtractor

## GraniteSpeechNarModel

[[autodoc]] GraniteSpeechNarModel
    - forward

## GraniteSpeechNarLanguageModel

[[autodoc]] GraniteSpeechNarLanguageModel
    - forward

## GraniteSpeechNarForCTC

[[autodoc]] GraniteSpeechNarForCTC
    - forward
    - generate

<!--Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# VibeVoice

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[VIbeVoice](https://hf.co/papers/2508.19205) is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.


**Core Architecture**

The VibeVoice framework integrates three key components:
1. **Continuous Speech Tokenizers:** Specialized acoustic and semantic tokenizers, where the acoustic tokenizer uses a $\sigma$-VAE to achieve ultra-low compression (7.5 tokens/sec, 3200x) for scalability and fidelity, and the semantic tokenizer uses an ASR proxy task for content-centric feature extraction.
2, **Large Language Model (LLM):** Use Qwen2.5 (in 1.5B and 7B versions) as its core sequence model.
3. **Token-Level Diffusion Head:** conditioned on the LLM's hidden state and responsible for predicting the continuous VAE features in a streaming fashion.

The original VibeVoice-1.5B checkpoint is available under the [Microsoft](https://hf.co/microsoft/VibeVoice-1.5B) organization on Hugging Face.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/bezzam/documentation-images/resolve/main/vibevoice_arch.png"/>
</div>


## Key Features

- **Long-Form Synthesis**: Can synthesize up to 90 minutes multi-speaker conversational speech.
- **Multi-Speaker Dialogue**: Capable of synthesizing audio with a maximum of 4 speakers.
- **State-of-the-Art Quality**: Outperforms baselines on both subjective and objective metrics.
- **High Compression**: Achieved by a novel acoustic tokenizer operating at an ultra-low 7.5 Hz frame rate.
- **Scalable LLM**: Scaling the core LLM from 1.5B to 7B significantly improves perceptual quality.


## Usage

### Batched Inference

### Pipeline Usage

### Making The Model Go Brrr

### Training


## VibeVoiceConfig

[[autodoc]] VibeVoiceConfig

## VibeVoiceDiffusionHeadConfig

[[autodoc]] VibeVoiceDiffusionHeadConfig

## VibeVoiceFeatureExtractor

[[autodoc]] VibeVoiceFeatureExtractor
    - __call__

## VibeVoiceProcessor

[[autodoc]] VibeVoiceProcessor
    - __call__

## VibeVoiceForConditionalGeneration

[[autodoc]] VibeVoiceForConditionalGeneration
    - forward
    - generate

## VibeVoiceModel

[[autodoc]] VibeVoiceModel

## VibeVoiceDiffusionHead

[[autodoc]] VibeVoiceDiffusionHead


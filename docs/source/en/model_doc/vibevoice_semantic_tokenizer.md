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
*This model was released on 2025-08-26 and added to Hugging Face Transformers on 2025-12-09.*

# VibeVoice Semantic Tokenizer

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[VibeVoice](https://huggingface.co/papers/2508.19205) is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.

One key feature of VibeVoice is the use of two continuous speech tokenizers, one for extracting [acoustic](./vibevoice_acoustic_tokenizer) features (this model) and another for semantic features (this model).

*Note: the semantic tokenizer can only be used to encode audio to extract semantic features.*

A model checkpoint is available at [bezzam/VibeVoice-SemanticTokenizer](https://huggingface.co/bezzam/VibeVoice-SemanticTokenizer)

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

## Architecture

The semantic tokenizer mirrors the hierarchical architecture of the [acoustic tokenizer](./vibevoice_acoustic_tokenizer)’s encoder, but without VAE components, as its objective is deterministic content-centric feature extraction. The main difference is the training objective, which uses Automatic Speech Recognition (ASR) as the proxy task. During training, its output is decoded by several Transformer decoder layers to predict text transcripts, aligning the semantic encoder’s representations with textual semantics. This decoder is discarded after pre-training. Please refer to the [technical report](https://huggingface.co/papers/2508.19205) for further details.


## Usage

Below is example usage to encode audio for extracting semantic features:

```python
import torch
from transformers import AutoFeatureExtractor, VibeVoiceSemanticTokenizerModel
from transformers.audio_utils import load_audio_librosa


model_id = "bezzam/VibeVoice-SemanticTokenizer"
sampling_rate = 24000

# load audio
audio = load_audio_librosa(
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
    sampling_rate=sampling_rate,
)

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = VibeVoiceSemanticTokenizerModel.from_pretrained(
    model_id, device_map=device,
).eval()

# preprocess audio
inputs = feature_extractor(
    audio,
    sampling_rate=sampling_rate,
    padding=True,
    pad_to_multiple_of=3200,
    return_attention_mask=False,
    return_tensors="pt",
).to(device)
print("Input audio shape:", inputs.input_features.shape)
# Input audio shape: torch.Size([1, 1, 224000])

# encode
with torch.no_grad():
    encoded_outputs = model.encode(inputs.input_features)
print("Latent shape:", encoded_outputs.latents.shape)
# Latent shape: torch.Size([1, 70, 128])
```


## VibeVoiceSemanticTokenizerConfig

[[autodoc]] VibeVoiceSemanticTokenizerConfig


## VibeVoiceSemanticTokenizerModel

[[autodoc]] VibeVoiceSemanticTokenizerModel
    - encode
    - forward

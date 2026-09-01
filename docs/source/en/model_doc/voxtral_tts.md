<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-03-26 and added to Hugging Face Transformers on 2026-03-31.*

# Voxtral TTS

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) is a multilingual zero-shot text-to-speech model from Mistral AI, proposed in [Voxtral TTS](https://huggingface.co/papers/2603.25551).

Voxtral TTS is a hybrid architecture made of 3 main components:

- **AR Backbone** (~3.4B params): A causal autoregressive decoder-only transformer (Ministral-3B-based) that takes tokenized text and optional voice reference audio codes as input, and predicts semantic audio tokens.
- **Flow-Matching Transformer** (~390M params): A 3-layer bidirectional transformer that predicts acoustic tokens via flow matching. It takes the backbone hidden states, a timestep embedding, and current noisy acoustic embeddings as input, and uses an Euler ODE solver to produce acoustic token predictions.
- **Voxtral Codec** (~150M params): A convolutional-transformer decoder that converts semantic + acoustic tokens into a 24kHz mono waveform. Only decoder weights are shipped (voice reference uses pre-computed embeddings).

The model supports 9 languages: English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi. It includes 20 preset voice embeddings for zero-shot voice cloning.

This model was contributed by [sachinsingh](https://huggingface.co/sachinsingh).
The original code can be found [here](https://github.com/vllm-project/vllm-omni).

## Usage example

```python
>>> from transformers import VoxtralTtsProcessor, VoxtralTtsForTextToSpeech

>>> processor = VoxtralTtsProcessor.from_pretrained("mistralai/Voxtral-4B-TTS-2603")
>>> model = VoxtralTtsForTextToSpeech.from_pretrained("mistralai/Voxtral-4B-TTS-2603")

>>> inputs = processor("Hello, my name is Voxtral!", voice_preset="neutral_female")
>>> output = model.generate(**inputs)
>>> processor.save_audio(output.audio[0], "output.wav")
```

### Multilingual speech

Voxtral TTS supports multilingual synthesis with language-specific voice presets:

```python
>>> # French
>>> inputs = processor("Bonjour, comment allez-vous?", voice_preset="fr_female")
>>> output = model.generate(**inputs)

>>> # German
>>> inputs = processor("Guten Tag, wie geht es Ihnen?", voice_preset="de_male")
>>> output = model.generate(**inputs)
```

### Generation parameters

The generation process supports several parameters to control audio quality:

```python
>>> output = model.generate(
...     **inputs,
...     max_new_tokens=500,    # max audio frames (at 12.5Hz frame rate)
...     temperature=0.7,       # sampling temperature for semantic tokens
...     top_k=50,              # top-k sampling
...     n_nfe=8,               # number of Euler ODE steps for flow matching
...     cfg_alpha=1.2,         # classifier-free guidance scale
... )
```

## VoxtralTtsConfig

[[autodoc]] VoxtralTtsConfig

## VoxtralTtsFlowMatchingConfig

[[autodoc]] VoxtralTtsFlowMatchingConfig

## VoxtralTtsCodecConfig

[[autodoc]] VoxtralTtsCodecConfig

## VoxtralTtsProcessor

[[autodoc]] VoxtralTtsProcessor
    - __call__

## VoxtralTtsForTextToSpeech

[[autodoc]] VoxtralTtsForTextToSpeech
    - forward
    - generate


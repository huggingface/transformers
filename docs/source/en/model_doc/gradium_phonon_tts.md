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
*This model was contributed to Hugging Face Transformers on 2026-10-20.*

# Gradium Phonon TTS

## Overview

Gradium Phonon TTS is [Gradium](https://gradium.ai)'s text-to-speech model, trained with the recipe of [Pocket TTS](./pocket_tts): a causal transformer reads a voice prompt and the text, then emits one continuous 32-dimensional audio latent per step at 12.5 Hz through a one-step flow head, and an end-of-speech head decides when the utterance is over. The latents are decoded to 24 kHz audio by a [Pocket TTS audio tokenizer](./pocket_tts_audio_tokenizer) held by the processor, Pocket TTS's [`PocketTTSProcessor`].

It differs from Pocket TTS in its voice conditioning: the voice prompt is made of 512-dimensional latents of a separate speaker codec, and a learnt speaker vector is added to every audio latent position. The speaker codec is not released yet, so the released checkpoint ships its voices as precomputed latents (`voices/`) and its audio tokenizer as a decoder only: voices cannot be cloned from a recording.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).

## Usage

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "eustlb/gradium-phonon-tts"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

voice = load_file(hf_hub_download(model_id, "voices/Freya.safetensors"))["audio_prompt_latents"]
# an empty `audio` item marks the voice, given as precomputed latents
conversation = [
    {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Hello world. I am Gradium's Phonon."}]}
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True,
    processor_kwargs={"audio_prompt_latents": [voice]},
).to(model.device)

outputs = model.generate(**inputs)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "phonon.wav")
```

The tokenizer ([`GradiumPhononTTSTokenizer`]) prepares the text the way Gradium's runtime does, in its `language` (English by default): typographic characters are folded, `@`, `+`, `=` and large numbers are spelled out in the language. Set `processor.tokenizer.language = "fr"` to read French text (`"de"`, `"es"` and `"pt"` are supported too).

## GradiumPhononTTSConfig

[[autodoc]] GradiumPhononTTSConfig

## GradiumPhononTTSFlowHeadConfig

[[autodoc]] GradiumPhononTTSFlowHeadConfig

## GradiumPhononTTSTokenizer

[[autodoc]] GradiumPhononTTSTokenizer
    - normalize_text

## GradiumPhononTTSModel

[[autodoc]] GradiumPhononTTSModel
    - forward

## GradiumPhononTTSForConditionalGeneration

[[autodoc]] GradiumPhononTTSForConditionalGeneration
    - forward
    - generate

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

Gradium Phonon TTS is [Gradium](https://gradium.ai)'s family of on-device text-to-speech models, trained with the recipe of [Pocket TTS](./pocket_tts): a causal transformer reads a voice prompt and the text, then emits one continuous 32-dimensional audio latent per step at 12.5 Hz through a one-step flow head, and an end-of-speech head decides when the utterance is over. The latents are decoded to 24 kHz audio by a [Pocket TTS audio tokenizer](./pocket_tts_audio_tokenizer) held by the processor, Pocket TTS's [`PocketTTSProcessor`].

It differs from Pocket TTS in its voice conditioning: the voice prompt is made of 512-dimensional latents of a separate speaker codec, and a learnt speaker vector is added to every audio latent position. The speaker codec is not released, so voices come as precomputed latents (each checkpoint's `default-voice.safetensors`, and the catalog of [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices)) and the audio tokenizer is a decoder only: voices cannot be cloned from a recording.

| Checkpoint | Parameters | Languages | License |
|---|---|---|---|
| [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) | 31M | English | Apache 2.0 |
| [gradium/phonon-40m-en](https://huggingface.co/gradium/phonon-40m-en) | 40M | English | CC BY-NC 4.0 |
| [gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr) | 40M | French | CC BY-NC 4.0 |
| [gradium/phonon-40m-de](https://huggingface.co/gradium/phonon-40m-de) | 40M | German | CC BY-NC 4.0 |
| [gradium/phonon-40m-es](https://huggingface.co/gradium/phonon-40m-es) | 40M | Spanish | CC BY-NC 4.0 |
| [gradium/phonon-40m-pt](https://huggingface.co/gradium/phonon-40m-pt) | 40M | Portuguese | CC BY-NC 4.0 |
| [gradium/phonon-40m-multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) | 40M | English, French, German, Spanish, Portuguese | CC BY-NC 4.0 |

They all load the same way: switching checkpoints means changing the repo id.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts).

## Usage tips

- Prepare inputs with [`~PocketTTSProcessor.apply_chat_template`]: one `user` message per text, an empty `{"type": "audio"}` item where the voice goes, the voice latents in `processor_kwargs={"audio_prompt_latents": [...]}`, and `add_generation_prompt=True`.
- The tokenizer does not spell out numbers, amounts, symbols or addresses, see [Text normalization](#text-normalization).
- `generate` returns `audio_latents` and `audio_latents_mask`; decode them with [`PocketTTSProcessor.decode`].
- Generation samples as Gradium's runtime does: `temperature` 0.3 (the variance of the flow's starting noise), and it stops shortly after the end-of-speech logit crosses `eos_threshold`. `do_sample=False` starts the flow from zero noise.
- The model loads in bf16, the dtype its weights are stored in; pass `dtype=torch.float32` on CPU to match Gradium's runtime there.
- Split long text with [`PocketTTSProcessor.split_text`], one conversation per chunk.

## Usage

### Voice

A voice is a tensor of latents `(num_frames, 512)`. Each checkpoint ships one (`default-voice.safetensors`), and [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices) has about a hundred more, in every language, each under its voice ID (`voices.jsonl` lists their names, languages and descriptions). Any voice works with any checkpoint.

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

model_id = "gradium/phonon-31m-en"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id, device_map="auto")


def load_voice(voice_id):
    """A voice of gradium/phonon-voices, as latents of shape (num_frames, 512)."""
    path = hf_hub_download("gradium/phonon-voices", f"voices/{voice_id}.safetensors")
    return load_file(path)["speaker_wavs"][0].T.float()


# Easton (en-us); the checkpoint's own voice: hf_hub_download(model_id, "default-voice.safetensors")
voice = load_voice("LMDIYE0Oa377Qh6Z")
# an empty `audio` item marks the voice
conversation = [
    {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Hello from Phonon."}]}
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

### Batches and long text

Batch several texts, each with its own voice, with long text split into chunks:

```python
text = (
    "Phonon is a small text-to-speech model that runs on your device. "
    "Long texts are split into sentences, and each chunk is one conversation of the batch."
)
chunks = processor.split_text(text)
texts = chunks + ["And this one is said by another voice."]
voices = [load_voice("LMDIYE0Oa377Qh6Z")] * len(chunks) + [load_voice("NbpkqMVS3CJeq2j8")]
conversations = [
    [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}] for text in texts
]
inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True,
    processor_kwargs={"audio_prompt_latents": voices},
).to(model.device)

outputs = model.generate(**inputs)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio([torch.cat(audio[: len(chunks)]), audio[-1]], ["long.wav", "other.wav"])
```

### Text normalization

The tokenizer normalizes what does not depend on the language, as Gradium's runtime does:

| Text | Tokenizer |
|---|---|
| `“Hello” — it’s here…` | `"Hello" it's here.` |
| `hello (world); see you 🚀` | `Hello, world, see you.` |

It does not spell out numbers, amounts, symbols or addresses, which depend on the language:

| Text | Tokenizer | Gradium's runtime |
|---|---|---|
| `It costs $1,234.56.` | `It costs $1,234.56.` | `It costs 1 thousand 234 dollars and 56 cents.` |
| `Write to jean@gradium.ai` | `Write to jean@gradium.ai.` | `Write to jean at gradium dot A-I.` |
| `It is 21°C` | `It is 21°C.` | `It is 21 degrees Celsius.` |

For those, use the text normalization of Gradium's runtime, [ptts](https://github.com/gradium-ai/xn-ptts), or write them out.

## GradiumPhononTTSConfig

[[autodoc]] GradiumPhononTTSConfig

## GradiumPhononTTSFlowHeadConfig

[[autodoc]] GradiumPhononTTSFlowHeadConfig

## GradiumPhononTTSModel

[[autodoc]] GradiumPhononTTSModel
    - forward

## GradiumPhononTTSForConditionalGeneration

[[autodoc]] GradiumPhononTTSForConditionalGeneration
    - forward
    - generate

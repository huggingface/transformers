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

# Pocket TTS

## Overview

Pocket TTS is [Kyutai](https://kyutai.org)'s 100M-parameter text-to-speech model, a [Continuous Audio Language Model](https://huggingface.co/papers/2509.06926) that runs faster than real time on CPU. A causal transformer emits one continuous audio latent per frame (12.5 Hz) through a one-step flow head, and an end-of-speech head decides when to stop.

The processor holds the [Pocket TTS audio tokenizer](./pocket_tts_audio_tokenizer), a Mimi-based codec that encodes voice prompts and decodes generated latents to 24 kHz audio. Predefined voices ship as KV caches (`voices/`).

Two flow heads are released: Lagrangian self-distillation (`flow_type="lsd"`, default) and drifting (`flow_type="drifting"`, `english_drifting_26-09`). Finetuning is supported, see [Training](#training).

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

## Usage tips

- Prepare inputs with [`~PocketTTSProcessor.apply_chat_template`] (one `user` message per text, `add_generation_prompt=True`). The tokenizer ([`PocketTTSTokenizer`]) prepares the text like the original, with the settings of the checkpoint's language.
- `generate` returns `audio_latents` and `audio_latents_mask`; decode them with [`PocketTTSProcessor.decode`].
- `max_new_tokens` defaults to a budget based on text length, and generation stops shortly after the end-of-speech logit crosses `eos_threshold`.
- `do_sample=False` matches the original at temperature 0. With sampling, `temperature` (default 0.3) is the variance of the flow's starting noise.
- Split long text with [`PocketTTSProcessor.split_text`], one conversation per chunk.

## Usage

### Voice

The voice can be given in three ways.

**1. Cloning a recording**: an `audio` item in the conversation.

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": url},
            {"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

outputs = model.generate(**inputs)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "cloned.wav")
```

**2. Precomputed latents**: an empty `audio` item, with the latents passed to the processor.

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

# e.g. a recording encoded once by the processor's audio tokenizer,
# to reuse without re-encoding it
url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
features = processor.feature_extractor(
    load_audio(url, sampling_rate=24000), sampling_rate=24000
).to(processor.audio_tokenizer.device)
# (num_frames, latent_dim)
voice = processor.audio_tokenizer.encode(**features).latents[0]

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio"},
            {"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."},
        ],
    }
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
processor.save_audio(audio, "latents.wav")
```

**3. Cached context**: the voice prompt's KV cache, passed as `past_key_values`. The shipped voices (`voices/`) use this form.

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts"
# cached voices are never encoded: the audio tokenizer's decoder is enough
processor = AutoProcessor.from_pretrained(model_id, encoder_config=None, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

voice = load_file(hf_hub_download(model_id, "voices/alba.safetensors"))
layers = []
for i, layer in enumerate(model.model.layers):
    # each layer's cache on that layer's device
    device = layer.self_attn.k_proj.weight.device
    layers.append(
        (voice[f"layers.{i}.keys"].to(device), voice[f"layers.{i}.values"].to(device))
    )
voice_cache = DynamicCache(layers, config=model.config)

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."}],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

outputs = model.generate(**inputs, past_key_values=voice_cache)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "cached.wav")
```

### Batches and long text

Batch several texts and voices, with long text split into chunks:

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts"
# cached voices are never encoded: the audio tokenizer's decoder is enough
processor = AutoProcessor.from_pretrained(model_id, encoder_config=None, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")


def load_voices(names):
    """The shipped voices, one per conversation, as one cache, left-padded to the
    longest voice, and the mask of their positions.

    `generate` extends the cache: load it for every call.
    """
    voices = [
        load_file(hf_hub_download(model_id, f"voices/{name}.safetensors"))
        for name in names
    ]
    lengths = [voice["layers.0.keys"].shape[2] for voice in voices]
    length = max(lengths)
    layers = []
    for i, layer in enumerate(model.model.layers):
        device = layer.self_attn.k_proj.weight.device
        keys, values = (
            torch.cat(
                [
                    torch.nn.functional.pad(voice[f"layers.{i}.{kind}"], (0, 0, length - n, 0))
                    for voice, n in zip(voices, lengths)
                ]
            ).to(device)
            for kind in ("keys", "values")
        )
        layers.append((keys, values))
    mask = torch.tensor([[0] * (length - n) + [1] * n for n in lengths])
    return DynamicCache(layers, config=model.config), mask


text = (
    "A first sentence that is long enough to be worth splitting. "
    "Then a second one, and a third one for good measure."
)
chunks = processor.split_text(text)
conversations = [
    [{"role": "user", "content": [{"type": "text", "text": chunk}]}]
    for chunk in chunks + ["Hello world"]
]
inputs = processor.apply_chat_template(
    conversations, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

voice_cache, voice_mask = load_voices(["alba"] * len(chunks) + ["jean"])
# the mask covers the cached voices too
inputs["attention_mask"] = torch.cat(
    [voice_mask.to(inputs["attention_mask"]), inputs["attention_mask"]], dim=-1
)
outputs = model.generate(**inputs, past_key_values=voice_cache)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(
    [torch.cat(audio[: len(chunks)]), audio[-1]], ["long.wav", "hello.wav"]
)
```

### Drifting checkpoint

Same usage, with the checkpoint's own voices:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts-english-drifting"
# cached voices are never encoded: the audio tokenizer's decoder is enough
processor = AutoProcessor.from_pretrained(model_id, encoder_config=None, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

voice = load_file(hf_hub_download(model_id, "voices/alba.safetensors"))
layers = []
for i, layer in enumerate(model.model.layers):
    device = layer.self_attn.k_proj.weight.device
    layers.append(
        (voice[f"layers.{i}.keys"].to(device), voice[f"layers.{i}.values"].to(device))
    )

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."}],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    **inputs, past_key_values=DynamicCache(layers, config=model.config)
)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "drifting.wav")
```

## Training

Put the target audio in the `assistant` turn. The processor returns its latents as `labels`, and the model returns the original's loss (LSD flow loss plus `config.eos_loss_weight` times the end-of-speech loss). The voice prompt is typically the beginning of the same utterance.

```python
import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")
model.train()

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": voice_prompt},
            {"type": "text", "text": target_text},
        ],
    },
    {"role": "assistant", "content": [{"type": "audio", "audio": target_audio}]},
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True
).to(model.device)
loss = model(**inputs).loss
loss.backward()
```

- Give the voice as a recording or latents, not a cache.
- For classifier-free guidance, drop the `audio` or `text` item of the `user` turn (the original drops each with probability 0.2).
- Precomputed latents go in `processor_kwargs={"audio_prompt_latents": [...], "labels": [...]}`, with empty `{"type": "audio"}` items as placeholders.
- Original recipe: AdamW, `learning_rate=2e-5` (2e-4 for a new language), `adam_beta2=0.95`, `weight_decay=0.1`, `max_grad_norm=1.0`, 1000 warmup steps then constant, EMA of the weights (decay 0.999).

## PocketTTSConfig

[[autodoc]] PocketTTSConfig

## PocketTTSFlowHeadConfig

[[autodoc]] PocketTTSFlowHeadConfig

## PocketTTSTokenizer

[[autodoc]] PocketTTSTokenizer
    - normalize_text

## PocketTTSFeatureExtractor

[[autodoc]] PocketTTSFeatureExtractor
    - __call__

## PocketTTSProcessor

[[autodoc]] PocketTTSProcessor
    - __call__
    - split_text
    - decode
    - save_audio

## PocketTTSModel

[[autodoc]] PocketTTSModel
    - forward

## PocketTTSForConditionalGeneration

[[autodoc]] PocketTTSForConditionalGeneration
    - forward
    - generate
    - sample_latent
    - denormalize_latents

## PocketTTSGenerateOutput

[[autodoc]] models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput

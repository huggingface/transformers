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
*This model was contributed to Hugging Face Transformers on 2026-10-12.*

# Pocket TTS

## Overview

Pocket TTS is [Kyutai](https://kyutai.org)'s 100M-parameter text-to-speech model, a [Continuous Audio Language Model](https://huggingface.co/papers/2509.06926) small enough to run faster than real time on a CPU. A causal transformer reads a voice prompt and the text, then emits one continuous 32-dimensional audio latent per step at 12.5 Hz through a one-step flow head: an MLP with adaptive layer norm that maps Gaussian noise to the latent. An end-of-speech head decides when the utterance is over.

The audio side is the [Pocket TTS audio tokenizer](./pocket_tts_audio_tokenizer), a Mimi-architecture codec with a continuous latent, held by the processor like a text tokenizer: it encodes a voice recording into the latents the model is prompted with, and decodes the generated latents into 24 kHz audio. Each checkpoint also ships its predefined voices the way the original does, as the KV cache of the voice prompt (`voices/`), passed to `generate` as `past_key_values`.

Two flow heads are released: Lagrangian self-distillation (`flow_type="lsd"`, the default English and other language checkpoints, conditioned on the start and end times of a flow step) and drifting (`flow_type="drifting"`, `english_drifting_26-09`, no time conditioning).

The model supports finetuning with the original's loss (Lagrangian self-distillation for the flow head, plus the end-of-speech loss), see [Training](#training).

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

## Usage tips

- Inputs go through [`~PocketTTSProcessor.apply_chat_template`], one `user` message per text: the chat template prepares the text the way the original does (capitalisation, final punctuation, the language's character replacements), and `add_generation_prompt=True` appends the position the first audio frame is generated from.
- `generate` returns the padded `audio_latents` and their `audio_latents_mask` (`return_dict_in_generate=True` in the checkpoints' generation config); [`PocketTTSProcessor.decode`] decodes them as one batch and fades in the start of each waveform as the original does (`fade_in_duration`, 5 ms).
- `generate` defaults `max_new_tokens` to the original's budget, `ceil((num_text_tokens / 3 + 2) * frame_rate)` frames for the longest text of the batch (`config.frame_rate`, 12.5 Hz), ignores the end-of-speech logit on the first `min_frames_before_eos` frames, and stops a few frames after it crosses `eos_threshold` (`frames_after_eos`, a processor output that depends on the text length).
- `do_sample=False` reproduces the original at temperature 0; with `do_sample=True`, `temperature` is the variance of the noise the flow starts from (the original's default, 0.3, is in the checkpoint's generation config). `num_flow_steps`, `eos_threshold` and `min_frames_before_eos` default to the generation config and can be overridden per call.
- The original splits long text into sentence-aligned chunks of at most 50 tokens and generates each one separately: [`PocketTTSProcessor.split_text`] does the same, one conversation per chunk.

## Usage

### Voice

The voice is given in one of three equivalent ways.

**1. Cloning a recording**: an `audio` item in the conversation, mono at 24 kHz, encoded by the processor's audio tokenizer (and ended on a short pause, `end_on_pause`, as the original does).

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

**2. Precomputed latents**: an empty `audio` item marks the voice, and its latents are passed to the processor, which writes one voice placeholder per latent frame.

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

**3. Cached context**: the backbone is causal, so the keys and values of the voice prompt do not depend on the text after it. They can be computed once and passed as `past_key_values`, the conversation then holding the text only. The checkpoint's voices (`voices/`) ship in this form, as the original's do; a cache of any voice is built the same way, by running the model on its voice prompt and keeping `past_key_values`.

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
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

Several texts and voices at once, one long text split into sentence chunks the way the original does, with one cached voice per conversation:

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
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

`english_drifting_26-09` is used the same way, with its own voices (a cache depends on the model's weights):

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts-english-drifting"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
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

A training example is a conversation whose `assistant` turn holds the target audio. The processor encodes the voice and the target in one batch of the audio tokenizer and returns the target latents as `labels`, the chat template keeps the text as written (the original trains on its transcripts and normalises only at inference), and the model returns the loss of the original's finetuning recipe: the LSD flow loss of the target frames plus `config.eos_loss_weight` times the end-of-speech loss. As in the original, the voice prompt is a recording of the same speaker, typically the beginning of the same utterance cut at a word boundary, and the target the rest of it with its transcript.

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

- The voice is given as a recording or as latents (ways 1 and 2 above), not as a cache: the original trains on the beginning of the same utterance. Training audio is encoded as is, with no pause appended to the voice (`end_on_pause`), as the original's data loader does.
- Classifier-free guidance is trained by dropping the voice and/or the text of a conversation (the original drops each with probability 0.2): leave out the `audio` item or the `text` item of the `user` turn.
- Precomputed latents are passed as `processor_kwargs={"audio_prompt_latents": [...], "labels": [...]}`, with an `{"type": "audio"}` item (no data) marking the voice in the `user` turn and the target in the `assistant` turn.
- The original's finetuning recipe is AdamW with `learning_rate=2e-5` (2e-4 for a new language), `adam_beta2=0.95`, `weight_decay=0.1`, `max_grad_norm=1.0`, a constant schedule after 1000 warmup steps, and an exponential moving average of the weights (decay 0.999) for the exported model.
- The latent statistics (`latent_mean`, `latent_std`) follow the training targets for the first `config.latent_stats_ema_steps` training forwards (1000), as in the original; set it to 0 to keep them.
- The self-distillation term of the LSD loss is computed on a fraction `config.lsd_distill_probability` (0.25) of the training steps, drawn from a generator seeded alike on every process.

## PocketTTSConfig

[[autodoc]] PocketTTSConfig

## PocketTTSFlowHeadConfig

[[autodoc]] PocketTTSFlowHeadConfig

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

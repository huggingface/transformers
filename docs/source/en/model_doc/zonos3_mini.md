<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-13.*

# ZONOS3-Mini

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
</div>

## Overview

ZONOS3-Mini is a multilingual text-to-speech model by Zyphra, introduced in the
[ZONOS3-Mini post](https://www.zyphra.com/post/zonos3-mini). It generates the codes of the [ZAC-1](./zac) codec at
15 frames per second, 44.1 kHz, and clones a voice from a short clip.

The model reads and writes rows of 11 columns: the 10 ZAC-1 codebooks, delayed by one row per codebook (codebook `j`
of frame `t` sits in row `t + j`), and a text column. A prompt is laid out as

`[voice] [3 condition rows] [BOS text EOS] [silence frames]`

and generation continues the silence with speech until every codebook has emitted the end-of-audio id.

| Part | Description |
|---|---|
| Language model | 11 layers of 1024 dims: gated DeltaNet layers, with a gated attention layer every 4 layers (16 heads, 4 KV heads, interleaved RoPE, unweighted QK norms with a learned temperature). The first feed-forward is dense; the others are 64-expert top-2 MoEs in a 256-dim latent space. Every sublayer reads a softmax mix over depth of the embeddings and all earlier sublayer outputs ("attention residuals"). |
| Heads | One softcapped head per codebook (4096 codes, end-of-audio and padding). |
| Speaker encoder | MECT-B2 (AntSpeaker) on Kaldi filterbanks at 16 kHz, pooled into a 192-dim speaker embedding. |
| Conditions | Loudness, fmax and speaking rate, each bucketed (or unknown) into a condition row. |
| Codec | [ZAC-1](./zac), held by the processor as `audio_tokenizer`. |

The voice comes from one of two places:

- **Reference** (default): the ZAC-1 codes of a clip (or of a released voice) prefix the prompt, and the model
  continues them.
- **Embedding**: the speaker encoder's embedding of a clip (or a released voice's embedding) fills a speaker row.

The checkpoint [Zyphra/ZONOS3-Mini-transformers](https://huggingface.co/Zyphra/ZONOS3-Mini-transformers) (TODO:
coordinate the final repo with Zyphra) ships 27 voices in `voices/`, each with its reference codes and embedding. The
original code can be found at [Zyphra/ZONOS3-Mini-hf](https://huggingface.co/Zyphra/ZONOS3-Mini-hf).

The processor normalizes the text (numbers, dates, currencies, units) with
[`zyphra_textnorm`](https://pypi.org/project/zyphra-textnorm/) when it is installed, as the model was trained; without
it, the text is used as is (line breaks collapsed).

## Usage

### A released voice

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "Zyphra/ZONOS3-Mini-transformers"  # TODO: move to final Zyphra repo
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

voice = load_file(hf_hub_download(model_id, "voices/Savannah.safetensors"))
conversation = [{"role": "user", "content": "Hello! This is ZONOS3-Mini running in transformers."}]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    processor_kwargs={"speaker_reference_codes": [voice["codes"]]},
).to(model.device)

sequences = model.generate(**inputs)
audio = processor.decode(sequences)
processor.save_audio(audio, "zonos3_mini.wav")
```

Pass `speaker_embeddings=voice["embedding"][None]` instead of `speaker_reference_codes` to condition on the voice's
speaker embedding. With reference codes, `processor.decode(sequences, speaker_reference_codes=[voice["codes"]])` also
decodes the last `decode_context_frames` reference frames before the speech and drops them, for a smoother start.

### Voice cloning from a clip

The voice clip goes in the user turn, at 44.1 kHz. In reference mode (the default), the processor's audio tokenizer
encodes it into reference codes:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This voice was cloned from a short clip."},
            {"type": "audio", "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, processor_kwargs={"sampling_rate": 44100}
).to(model.device)
audio = processor.decode(model.generate(**inputs))
```

In embedding mode, the speaker encoder embeds the clip into the speaker row:

```python
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    processor_kwargs={"sampling_rate": 44100, "speaker_mode": "embedding"},
).to(model.device)
audio = processor.decode(model.generate(**inputs))
```

### Speaker guidance

Classifier-free guidance on the voice strengthens the cloning: the processor also returns the prompt without its
voice, and `generate` combines both with `guidance_scale` (which must be greater than 1).

```python
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    processor_kwargs={"sampling_rate": 44100, "speaker_guidance": True},
).to(model.device)
audio = processor.decode(model.generate(**inputs, guidance_scale=1.5))
```

### Conditions and sampling

Loudness (LUFS), fmax (Hz) and speaking rate (UTF-8 bytes per second) can be requested as values (bucketed with the
checkpoint's edges) or buckets; unspecified conditions are unknown. The release's sampling defaults (temperature 0.75,
top-k 80, top-p 0.69, a repetition penalty of 2.4 over the last 96 rows on the first 6 codebooks) are in the
generation config, and can be overridden in `generate`:

```python
inputs = processor(
    text="A little faster, please.",
    speaker_reference_codes=[voice["codes"]],
    condition_values={"speaking_rate": 16.0},
    return_tensors="pt",
).to(model.device)
sequences = model.generate(**inputs, temperature=0.6, repetition_window=64)
```

### Batched inference

Prompts are left-padded to the same length:

```python
texts = ["The first sentence.", "A second, somewhat longer sentence to speak."]
inputs = processor(
    text=texts, speaker_reference_codes=[voice["codes"], voice["codes"]], return_tensors="pt"
).to(model.device)
sequences = model.generate(**inputs)
audios = processor.decode(sequences)
```

### Pipeline

```python
from transformers import pipeline

pipe = pipeline("text-to-speech", model=model_id)
speech = pipe("Hello from the pipeline.")
# speech["audio"], speech["sampling_rate"]
```

### `torch.compile`

The language model compiles to a full graph, so `generate` can run with a static cache and a compiled forward:

```python
sequences = model.generate(**inputs, cache_implementation="static")
```

The codec decoder also runs on fixed-size tensors and compiles:

```python
processor.audio_tokenizer.decode = torch.compile(processor.audio_tokenizer.decode, fullgraph=True)
audio = processor.decode(sequences)
```

### Training

A training conversation adds an `assistant` turn holding the target audio. With `output_labels=True`, the processor
encodes it, appends its delayed rows to the prompt and returns them as `labels` (`-100` elsewhere); the loss is the
cross-entropy of each row's audio columns predicted from the previous row.

```python
conversation = [
    {"role": "user", "content": "The weather is lovely today."},
    {"role": "assistant", "content": [{"type": "audio", "path": "target.wav"}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    processor_kwargs={"sampling_rate": 44100, "output_labels": True},
).to(model.device)
loss = model(**inputs).loss
loss.backward()
```

## Zonos3MiniConfig

[[autodoc]] Zonos3MiniConfig

## Zonos3MiniSpeakerEncoderConfig

[[autodoc]] Zonos3MiniSpeakerEncoderConfig

## Zonos3MiniFeatureExtractor

[[autodoc]] Zonos3MiniFeatureExtractor
    - __call__

## Zonos3MiniProcessor

[[autodoc]] Zonos3MiniProcessor
    - __call__
    - decode
    - encode_reference
    - normalize_text
    - save_audio

## Zonos3MiniSpeakerEncoderModel

[[autodoc]] Zonos3MiniSpeakerEncoderModel
    - forward

## Zonos3MiniModel

[[autodoc]] Zonos3MiniModel
    - forward

## Zonos3MiniForConditionalGeneration

[[autodoc]] Zonos3MiniForConditionalGeneration
    - forward
    - generate

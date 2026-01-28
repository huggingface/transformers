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

*This model was released on 2025-11-03 and added to Hugging Face Transformers on 2026-01-08.*

# Music Flamingo

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Music Flamingo is a fully open large audio–language model designed for robust understanding and reasoning over music. It builds upon Audio Flamingo 3 architecture, pairing a Whisper-style audio encoder with a causal language model and performing replace-in-place audio–text fusion: the processor aligns post-pool audio frames to a dedicated placeholder token and the model replaces those token slots with projected audio embeddings during the forward pass.

The model checkpoint is available at: [nvidia/music-flamingo-2601-hf](https://huggingface.co/nvidia/music-flamingo-2601-hf)

Highlights:

- Unified audio encoder across speech, sound, and music.
- **Rotary Time Embeddings (RoTE)** for enhanced temporal modeling, enabling support for **up to 20 minutes of audio**.
- **Extended long-audio support via windowing and post-pool alignment (up to 20 minutes maximum).** The model processes audio in 30-second windows with a hard limit of 40 windows (20 minutes total). Audio longer than 20 minutes will be truncated.
- Special sound boundary tokens (`<|sound_bos|>` and `<|sound_eos|>`) for improved audio sequence modeling.
- Deterministic fusion that preserves sequence length by replacing audio placeholder tokens with audio embeddings.

This model was contributed by [Lasha Koroshinadze](https://huggingface.co/lashahub) and [Eric Bezzam](https://huggingface.co/bezzam).

### Paper

[Music Flamingo: Scaling Music Understanding in Audio Language Models](https://huggingface.co/papers/2511.10289)  
S. Ghosh, A. Goel, L. Koroshinadze, S. Lee, Z. Kong, J. F. Santos, R. Duraiswami, D. Manocha, W. Ping, M. Shoeybi, B. Catanzaro  
NVIDIA and University of Maryland  
Project: https://research.nvidia.com/labs/adlr/MF/

## Usage

### Audio Instruct Mode

The model supports audio-text instructions, including multi-turn interactions, all processed in batches.

➡️ audio + text instruction

```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."},
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ multi-turn:

```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Write a rich caption that blends the technical details (genre, BPM, key, chords, mix) with how the song feels emotionally and dynamically as it unfolds.",
            },
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This energetic Eurodance anthem at 150 BPM in E major combines bright synth arpeggios with a punchy four-on-the-floor beat..."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What instruments stand out the most?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ text only:

```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ audio only:

```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_2.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ batched inference!

```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Generate a structured lyric sheet from the input music.",
                },
                {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_2.mp3"},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ Training:

```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.train()

conversation = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Break the track down like a critic - list its tempo, key, and chordal motion, then explain the textures, dynamics, and emotional impact of the performance."},
                {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This Eurodance track operates at 150 BPM in E major, with harmonic movement centering on the I-vi-IV-V family. The production features layered synth arpeggios, a four-on-the-floor kick pattern, and a mezzo-soprano lead vocal with bright timbre. Dynamically, the track builds through verses into an anthemic chorus with full synth orchestration and backing vocals, creating an uplifting, euphoric atmosphere characteristic of late 2000s dance-pop."}],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this song from both a technical and artistic lens: mention tempo, harmony, and instrumentation, but also mood, lyrical themes, and structure.",
                },
                {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_2.mp3"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This electronic pop track combines upbeat production with playful lyrical themes centered around late-night pizza cravings. The structure follows a verse-chorus format with recurring melodic motifs and rhythmic patterns that emphasize the celebratory, lighthearted mood of the piece."}],
        }

    ]
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    output_labels=True,
).to(model.device)

loss = model(**inputs).loss
loss.backward()
```

## How the model works

### Architecture

* **MusicFlamingoEncoder**
  Whisper-style feature extractor + encoder with **Rotary Time Embeddings (RoTE)** → average-pool over time (stride 2) → LayerNorm.
  Produces per-frame hidden states at the post-pool rate. RoTE enables the model to handle temporal information for audio sequences up to 20 minutes.

* **MusicFlamingoMultiModalProjector**
  A small MLP that maps encoder features to the language model's hidden size.

* **MusicFlamingoForConditionalGeneration**
  A causal language model that accepts text embeddings where each audio placeholder token slot is replaced, in place, by an audio frame embedding. Uses special boundary tokens (`<|sound_bos|>` and `<|sound_eos|>`) to mark audio sequences. No sequence-length change is introduced by fusion.

### Processor-level alignment

1. Each raw waveform is split into fixed-length windows based on the feature extractor’s `chunk_length` (seconds) and `sampling_rate` (Hz).
2. For each window, the processor computes the number of post-pool frames `post_pool_len` that the encoder will output (matching the conv/pool schedule).
3. The processor expands the audio placeholder token by the total number of post-pool frames across all windows.
4. The model later replaces those token positions with the corresponding projected audio embeddings.

## Long audio and windowing

**Important: Maximum audio length is 20 minutes.** Audio longer than this will be truncated.

* The default setup processes 30-second windows at 16 kHz mono.
* **The processor enforces a hard limit of 40 windows per sample, resulting in a maximum of 20 minutes of audio (40 windows × 30 seconds).**
* Rotary Time Embeddings (RoTE) provide position information for sequences up to 20 minutes (1200 seconds).
* For each window:

  * `mel_len` is the padded mel length.
  * A conv stack reduces time as `conv_output_len = (mel_len - 1) // 2 + 1`.
  * Post-pool frames per window: `post_pool_len = (conv_output_len - 2) // 2 + 1`.
  * An audio placeholder token is expanded to the sum of `post_pool_len` across all windows.

## Padding, attention, and caching

* **Left padding vs right padding**
  For generation with mixed prompt lengths in a batch, left padding is usually preferable.
  For training, right padding is common; Music Flamingo's fusion mechanism itself is padding-agnostic because it replaces in place.
* **Attention masks**
  The processor returns `attention_mask` (text) and `input_features_mask` (audio). The model builds an internal 4-D mask on the encoder's pre-pool axis with negative infinity at pad positions.
* **Audio boundary tokens**
  The model uses special tokens `<|sound_bos|>` and `<|sound_eos|>` to explicitly mark the beginning and end of audio sequences.
* **Caching**
  During generation, `input_features` and `input_features_mask` are only passed on the first step. Subsequent steps use cached keys/values from the language model.

## Troubleshooting

* Empty or truncated outputs when batching
  Use left padding for batched generation and decode only the new tokens after the prompt length, as shown in the quickstart.

## MusicFlamingoConfig

[[autodoc]] MusicFlamingoConfig

## MusicFlamingoEncoderConfig

[[autodoc]] MusicFlamingoEncoderConfig

## MusicFlamingoProcessor

[[autodoc]] MusicFlamingoProcessor

## MusicFlamingoEncoder

[[autodoc]] MusicFlamingoEncoder
    - forward

## MusicFlamingoForConditionalGeneration

[[autodoc]] MusicFlamingoForConditionalGeneration
    - forward

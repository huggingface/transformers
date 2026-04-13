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

*This model was released on 2026-04-13 and added to Hugging Face Transformers on 2026-04-13.*

# Audio Flamingo Next

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Audio Flamingo Next (AF-Next) is a fully open large audio-language model for understanding and reasoning over speech, environmental sounds, and music. In Transformers, it is implemented as a replace-in-place audio-text model: the processor expands a dedicated audio placeholder into the exact number of post-pool audio frames, and the model replaces those token slots with projected audio embeddings during the forward pass.

AF-Next builds on [Audio Flamingo 3](./audioflamingo3.md) with a stronger audio-language training recipe, Rotary Time Embeddings (RoTE) for temporally grounded modeling, and support for long and complex audio inputs up to **30 minutes (1800 seconds)**.

The model checkpoint is available at: [nvidia/audio-flamingo-next-hf](https://huggingface.co/nvidia/audio-flamingo-next-hf)

Highlights:

- Generalist audio understanding across speech, sound, and music.
- **RoTE-based temporal grounding** for long-form audio reasoning.
- **Long-audio support up to 30 minutes** through 30-second windowing and post-pool alignment.
- Special sound boundary tokens (`<|sound_bos|>` and `<|sound_eos|>`) for audio sequence boundaries.
- Deterministic replace-in-place fusion that preserves the text sequence length.

This model was contributed by [Lasha Koroshinadze](https://huggingface.co/lashahub) and [Eric Bezzam](https://huggingface.co/bezzam).

### Paper

Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music  
Sreyan Ghosh, Arushi Goel, Kaousheik Jayakumar, Lasha Koroshinadze, Nishit Anand, Zhifeng Kong, Siddharth Gururani, Sang-gil Lee, Jaehyeon Kim, Aya Aljafari, Chao-Han Huck Yang, Sungwon Kim, Ramani Duraiswami, Dinesh Manocha, Mohammad Shoeybi, Bryan Catanzaro, Ming-Yu Liu, Wei Ping  
NVIDIA and University of Maryland

The paper introduces AF-Next as an open frontier audio-language model trained with a curriculum spanning pre-training, mid-training, post-training, and time-grounded reasoning. Compared to Audio Flamingo 3, it expands long-audio support to 30 minutes, strengthens performance across speech, sound, and music tasks, and introduces Temporal Audio Chain-of-Thought for timestamp-grounded reasoning over long recordings.

## Usage

### Audio Instruct Mode

The model supports audio-text instructions, including multi-turn and batched interactions.

➡️ audio + text instruction

```python
from transformers import AudioFlamingoNextForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-next-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingoNextForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Transcribe the input speech.",
            },
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)
inputs["input_features"] = inputs["input_features"].to(model.dtype)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ multi-turn:

```python
from transformers import AudioFlamingoNextForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-next-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingoNextForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Transcribe the input speech.",
            },
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Summer follows spring, the days grow longer, and the nights are warm."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize the content in one sentence."},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)
inputs["input_features"] = inputs["input_features"].to(model.dtype)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ batched inference!

```python
from transformers import AudioFlamingoNextForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-next-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingoNextForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversations = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe the input speech.",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3",
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
                    "text": "Compose a detailed caption integrating all audio elements, describing sound effects, speech, and music.",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
                },
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
inputs["input_features"] = inputs["input_features"].to(model.dtype)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

➡️ Training:

```python
from transformers import AudioFlamingoNextForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-next-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingoNextForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.train()

conversation = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the input speech."},
                {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"},
            ],
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": "Summer follows spring, the days grow longer, and the nights are warm."}],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Compose a detailed caption integrating all audio elements, describing sound effects, speech, and music.",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
                },
            ],
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": "A low-fidelity recording begins with a brief, sharp electronic buzz, followed by high-energy electronic dance music with a driving beat, pulsing synth bass, and a bright repetitive melody. Over the music, a small dog's sharp barks cut through repeatedly, creating a lively and chaotic soundscape."}],
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
inputs["input_features"] = inputs["input_features"].to(model.dtype)

loss = model(**inputs).loss
loss.backward()
```

## How the model works

### Architecture

* **Audio Encoder**
  AF-Next uses the AF-Whisper encoder. Audio is resampled to 16 kHz mono, converted to a 128-bin log-mel spectrogram with a 25 ms window and 10 ms hop, encoded in non-overlapping 30-second windows, then pooled with stride 2.

* **Rotary Time Embeddings (RoTE)**
  AF-Next uses axial rotary embeddings over the window index and the encoder time index, and both axes are modulated with absolute timestamps in seconds. This is the same RoTE formulation used by Music Flamingo.

* **AudioFlamingoNextMultiModalProjector**
  A 2-layer MLP that maps AF-Whisper features to the language model hidden size.

* **AudioFlamingoNextForConditionalGeneration**
  A decoder-only audio-language model built on a Qwen2.5-7B text backbone. Audio placeholder token slots are replaced in place by projected audio frame embeddings, with `<|sound_bos|>` and `<|sound_eos|>` marking audio boundaries. The paper introduces multiple AF-Next variants; the current Transformers checkpoint exposes the public conditional-generation model in this standard interface.

### Processor-level alignment

1. Each raw waveform is resampled and split into fixed 30-second windows using the feature extractor configuration.
2. For each window, the processor computes the number of post-pool frames `post_pool_len` that the encoder will output (matching the conv/pool schedule).
3. The processor expands the audio placeholder token by the total number of post-pool frames across all windows.
4. The model later replaces those token positions with the corresponding projected audio embeddings.

## Long audio and windowing

**Important: Maximum audio length is 30 minutes.** Audio longer than this will be truncated.

* The default setup processes 30-second windows at 16 kHz mono.
* **The processor enforces a hard limit of 60 windows per sample, resulting in a maximum of 30 minutes of audio (60 windows × 30 seconds).**
* RoTE provides temporal position information for audio sequences up to 30 minutes (1800 seconds).
* For each window:

  * `mel_len` is the padded mel length.
  * A conv stack reduces time as `conv_output_len = (mel_len - 1) // 2 + 1`.
  * Post-pool frames per window: `post_pool_len = (conv_output_len - 2) // 2 + 1`.
  * An audio placeholder token is expanded to the sum of `post_pool_len` across all windows.

## AudioFlamingoNextConfig

[[autodoc]] AudioFlamingoNextConfig

## AudioFlamingoNextProcessor

[[autodoc]] AudioFlamingoNextProcessor

## AudioFlamingoNextForConditionalGeneration

[[autodoc]] AudioFlamingoNextForConditionalGeneration
    - forward

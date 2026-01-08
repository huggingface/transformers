<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2025-07-10 and added to Hugging Face Transformers on 2025-11-17.*

# Audio Flamingo 3

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Audio Flamingo 3 (AF3) is a fully open large audio–language model designed for robust understanding and reasoning over speech, environmental sounds, and music. AF3 pairs a Whisper-style audio encoder with a causal language model and performs replace-in-place audio–text fusion: the processor aligns post-pool audio frames to a dedicated placeholder token and the model replaces those token slots with projected audio embeddings during the forward pass.

The model checkpoint is available at: [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf)

Highlights:

- Unified audio encoder across speech, sound, and music.
- **Long-audio support via windowing and post-pool alignment (up to 10 minutes maximum).** The model processes audio in 30-second windows with a hard limit of 20 windows (10 minutes total). Audio longer than 10 minutes will be truncated.
- Deterministic fusion that preserves sequence length by replacing audio placeholder tokens with audio embeddings.

This model was contributed by [Lasha Koroshinadze](https://huggingface.co/lashahub) and [Eric Bezzam](https://huggingface.co/bezzam).

### Paper

[Audio Flamingo 3](https://huggingface.co/papers/2507.08128): Advancing Audio Intelligence with Fully Open Large Audio Language Models  
A. Goel, S. Ghosh, J. Kim, S. Kumar, Z. Kong, S. Lee, C.-H. H. Yang, R. Duraiswami, D. Manocha, R. Valle, B. Catanzaro  
NVIDIA and University of Maryland  
Project: https://research.nvidia.com/labs/adlr/AF3/

## Usage

### Audio Instruct Mode

The model supports audio-text instructions, including multi-turn interactions, all processed in batches.

➡️ audio + text instruction

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the input speech."},
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"},
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
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Instruction: How does the tone of female speech change throughout the audio? Choose the correct option among the options below: (A) Sad to happy (B) Happy to sad (C) Neutral to happy (D) Happy to neutral.",
            },
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/000000786159.31.wav"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "(A) Sad to happy"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Why do you think so?"},
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
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

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
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"},
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
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the input speech."},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav",
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
                    "text": "This track feels really peaceful and introspective. What elements make it feel so calming and meditative?",
                },
                {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/FPSbCAANfbJLVSwD.mp3"},
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
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
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
            "content": [{"type": "text", "text": "The transcription of the audio is 'summer follows spring the days grow longer and the nights are warm'."}],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This track feels really peaceful and introspective. What elements make it feel so calming and meditative?",
                },
                {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/FPSbCAANfbJLVSwD.mp3"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The transcription of the audio is 'some transcription of the audio'."}],
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

➡️ transcription shortcut

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(audio="https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, strip_prefix=True)

print(decoded_outputs)
```

The model is trained to emit transcriptions prefixed with assistant framing such as `The spoken content of the audio is "<text>".`. Use `strip_prefix=True` (as shown above) to remove the fixed assistant sentence and surrounding quotes so that only the transcription remains.

## How the model works

### Architecture

* **AudioFlamingo3Encoder**
  Whisper-style feature extractor + encoder → average-pool over time (stride 2) → LayerNorm.
  Produces per-frame hidden states at the post-pool rate.

* **AudioFlamingo3MultiModalProjector**
  A small MLP that maps encoder features to the language model’s hidden size.

* **AudioFlamingo3ForConditionalGeneration**
  A causal language model that accepts text embeddings where each audio placeholder token slot is replaced, in place, by an audio frame embedding. No sequence-length change is introduced by fusion.

### Processor-level alignment

1. Each raw waveform is split into fixed-length windows based on the feature extractor’s `chunk_length` (seconds) and `sampling_rate` (Hz).
2. For each window, the processor computes the number of post-pool frames `post_pool_len` that the encoder will output (matching the conv/pool schedule).
3. The processor expands the audio placeholder token by the total number of post-pool frames across all windows.
4. The model later replaces those token positions with the corresponding projected audio embeddings.

## Usage patterns

### Transcription shortcut

For automatic speech recognition you can skip writing the default instruction each time and call
[`~transformers.AudioFlamingo3Processor.apply_transcription_request`]:

```python
inputs = processor.apply_transcription_request(audio=audio_array)
```

Pass `prompt="Transcribe the input speech."` (or a list of prompts for batch audio) to customize the instruction while
keeping the audio placeholder handling.

`audio` accepts in-memory arrays, local file paths, or URLs. Any processor kwargs (`text_kwargs`, `audio_kwargs`, etc.)
are forwarded, so you can tweak padding or tensor formats just like when calling `processor(...)`.

## Long audio and windowing

**Important: Maximum audio length is 10 minutes.** Audio longer than this will be truncated.

* The default setup processes 30-second windows at 16 kHz mono.
* **The processor enforces a hard limit of 20 windows per sample, resulting in a maximum of 10 minutes of audio (20 windows × 30 seconds).**
* For each window:

  * `mel_len` is the padded mel length.
  * A conv stack reduces time as `conv_output_len = (mel_len - 1) // 2 + 1`.
  * Post-pool frames per window: `post_pool_len = (conv_output_len - 2) // 2 + 1`.
  * An audio placeholder token is expanded to the sum of `post_pool_len` across all windows.

## Padding, attention, and caching

* **Left padding vs right padding**
  For generation with mixed prompt lengths in a batch, left padding is usually preferable.
  For training, right padding is common; AF3’s fusion mechanism itself is padding-agnostic because it replaces in place.
* **Attention masks**
  The processor returns `attention_mask` (text) and `input_features_mask` (audio). The model builds an internal 4-D mask on the encoder’s pre-pool axis with negative infinity at pad positions.
* **Caching**
  During generation, `input_features` and `input_features_mask` are only passed on the first step. Subsequent steps use cached keys/values from the language model.

## Troubleshooting

* Empty or truncated outputs when batching
  Use left padding for batched generation and decode only the new tokens after the prompt length, as shown in the quickstart.

## AudioFlamingo3Config

[[autodoc]] AudioFlamingo3Config

## AudioFlamingo3EncoderConfig

[[autodoc]] AudioFlamingo3EncoderConfig

## AudioFlamingo3Processor

[[autodoc]] AudioFlamingo3Processor

## AudioFlamingo3Encoder

[[autodoc]] AudioFlamingo3Encoder
    - forward

## AudioFlamingo3ForConditionalGeneration

[[autodoc]] AudioFlamingo3ForConditionalGeneration
    - forward

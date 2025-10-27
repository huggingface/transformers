*This model was released on 2025-07-10 and added to Hugging Face Transformers on 2025-10-15.*

# Audio Flamingo 3

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Audio Flamingo 3 (AF3) is a fully open large audio–language model designed for robust understanding and reasoning over speech, environmental sounds, and music. AF3 pairs a Whisper-style audio encoder with a causal language model and performs replace-in-place audio–text fusion: the processor aligns post-pool audio frames to a dedicated placeholder token (written as `<sound>` in code) and the model replaces those token slots with projected audio embeddings during the forward pass.

Highlights:

- Unified audio encoder across speech, sound, and music.
- Long-audio support via windowing and post-pool alignment (up to 10 minutes).
- Deterministic fusion that preserves sequence length by replacing `<sound>` tokens with audio embeddings.

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

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the input speech."},
            {"type": "audio", "path": "https://audioflamingo3.github.io/static/chat/WhDJDIviAOg_120_10.mp3"},
        ],
    }
]

batch = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

gen_ids = model.generate(**batch, max_new_tokens=512)

inp_len = batch["input_ids"].shape[1]
new_tokens = gen_ids[:, inp_len:]
texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
print(texts)
```

➡️ multi-turn:

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Instruction: How does the tone of female speech change throughout the audio? Choose the correct option among the options below: (A) Sad to happy (B) Happy to sad (C) Neutral to happy (D) Happy to neutral.",
            },
            {"type": "audio", "path": "https://audioflamingo3.github.io/static/long_audio/000000786159.31.wav"},
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

batch = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

gen_ids = model.generate(**batch, max_new_tokens=512)

inp_len = batch["input_ids"].shape[1]
new_tokens = gen_ids[:, inp_len:]
texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
print(texts)
```

➡️ text only:

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"},
        ],
    }
]

batch = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

gen_ids = model.generate(**batch, max_new_tokens=512)

inp_len = batch["input_ids"].shape[1]
new_tokens = gen_ids[:, inp_len:]
texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
print(texts)
```

➡️ audio only:

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": "https://audioflamingo3.github.io/static/chat/WhDJDIviAOg_120_10.mp3"},
        ],
    }
]

batch = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

gen_ids = model.generate(**batch, max_new_tokens=512)

inp_len = batch["input_ids"].shape[1]
new_tokens = gen_ids[:, inp_len:]
texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
print(texts)
```

➡️ batched inference!

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the input speech."},
                {
                    "type": "audio",
                    "path": "https://audioflamingo3.github.io/static/long_speech/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav",
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
                {"type": "audio", "path": "https://audioflamingo3.github.io/static/chat/FPSbCAANfbJLVSwD.mp3"},
            ],
        }
    ],
]

batch = processor.apply_chat_template(
    conversations,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

gen_ids = model.generate(**batch, max_new_tokens=512)

inp_len = batch["input_ids"].shape[1]
new_tokens = gen_ids[:, inp_len:]
texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
print(texts)
```

## How the model works

### Architecture

* **AudioFlamingo3Encoder**
  Whisper-style feature extractor + encoder → average-pool over time (stride 2) → LayerNorm.
  Produces per-frame hidden states at the post-pool rate.

* **AudioFlamingo3MultiModalProjector**
  A small MLP that maps encoder features to the language model’s hidden size.

* **AudioFlamingo3ForConditionalGeneration**
  A causal language model that accepts text embeddings where each `<sound>` token slot is replaced, in place, by an audio frame embedding. No sequence-length change is introduced by fusion.

### Processor-level alignment

1. Each raw waveform is split into fixed-length windows based on the feature extractor’s `chunk_length` (seconds) and `sampling_rate` (Hz).
2. For each window, the processor computes the number of post-pool frames `K` that the encoder will output (matching the conv/pool schedule).
3. The processor expands the audio placeholder token by the total number of post-pool frames across all windows.

   * If the prompt contains no `<sound>`, all expanded tokens are inserted at the start of the user message in the chat template (or prepended to plain text if no template is used).
   * If the prompt contains `<sound>`, exactly one placeholder is supported and it is expanded in place. Multiple placeholders are not supported.
4. The model later replaces those token positions with the corresponding projected audio embeddings.

This design guarantees a 1:1 match between placeholder positions in the text and encoder frame outputs, enabling safe batching without ragged audio–text concatenation.

## Usage patterns

### Single-turn prompts (recommended)

You can omit `<sound>` entirely and let the processor insert the expanded tokens:

```python
prompt = "Transcribe the input speech."
inputs = processor(text=prompt, audio=audio_array)
```

### Explicit placeholder control (advanced)

If you include `<sound>` in your prompt, provide exactly one placeholder; it will be expanded to match the total number of post‑pool frames across all windows:

```python
# For a clip that may span multiple windows:
prompt = "<sound>\nDescribe the audio in detail."
inputs = processor(text=prompt, audio=audio_array)
```

Notes:

* The processor computes window counts automatically and expands a single `<sound>` to the correct total number of frame tokens.
* Multiple `<sound>` placeholders are not supported and will raise an error.

## Long audio and windowing

* The feature extractor’s `chunk_length` and `sampling_rate` determine window size.
  The default setup processes approximately 30-second windows at 16 kHz mono.
* The processor caps the total number of windows per sample to a practical limit (about 10 minutes by default).
* For each window:

  * `L_mel` is the padded mel length.
  * A conv stack reduces time as `L1 = (L_mel - 1) // 2 + 1`.
  * Post-pool frames per window: `K = (L1 - 2) // 2 + 1`.
  * A single `<sound>` placeholder is expanded to the sum of `K` across all windows.

## Padding, attention, and caching

* **Left padding vs right padding**
  For generation with mixed prompt lengths in a batch, left padding is usually preferable.
  For training, right padding is common; AF3’s fusion mechanism itself is padding-agnostic because it replaces in place.
* **Attention masks**
  The processor returns `attention_mask` (text) and `feature_attention_mask` (audio). The model builds an internal 4-D mask on the encoder’s pre-pool axis with negative infinity at pad positions.
* **Caching**
  During generation, `input_features` and `feature_attention_mask` are only passed on the first step. Subsequent steps use cached keys/values from the language model.

## Troubleshooting

* Error: “Audio tokens and features mismatch”
  Cause: The number of `<sound>` tokens in `input_ids` does not match the total number of post-pool frames.
  Fix: Let the processor handle expansion. If you include `<sound>` explicitly, use exactly one placeholder.

* Error: “Sample X: found N '<sound>' placeholders. Expected exactly 1 or 0 placeholders.”
  Cause: Multiple placeholders are not supported.
  Fix: Remove extra placeholders or omit them entirely and let the processor insert them.

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

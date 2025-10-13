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
- Production-oriented processing flow with batch safety and strict shape checking.

### Paper

[Audio Flamingo 3](https://huggingface.co/papers/2507.08128): Advancing Audio Intelligence with Fully Open Large Audio Language Models  
A. Goel, S. Ghosh, J. Kim, S. Kumar, Z. Kong, S. Lee, C.-H. H. Yang, R. Duraiswami, D. Manocha, R. Valle, B. Catanzaro  
NVIDIA and University of Maryland  
Project: https://research.nvidia.com/labs/adlr/AF3/

## Quickstart

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto").eval()

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the input speech."},
                {"type": "audio", "path": "audio_1.wav"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the song."},
                {"type": "audio", "path": "audio_2.wav"},
            ],
        }
    ]
]

batch = processor.apply_chat_template(
    conversations,
    tokenize=True,
    add_generation_prompt=True,
    sampling_rate=getattr(processor.feature_extractor, "sampling_rate", 16000),
    return_dict=True,
).to(model.device)

gen_ids = model.generate(**batch, max_new_tokens=512)

inp_len = batch["input_ids"].shape[1]
new_tokens = gen_ids[:, inp_len:]
texts = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
3. The processor expands the audio placeholder token `<sound>` exactly `K` times per window.

   * If the prompt contains no `<sound>`, all expanded tokens are **prepended** to the text.
   * If the prompt contains `<sound>`, the number of placeholders must equal the number of windows.
4. The model later replaces those token positions with the corresponding projected audio embeddings.

This design guarantees a 1:1 match between placeholder positions in the text and encoder frame outputs, enabling safe batching without ragged audio–text concatenation.

## Usage patterns

### Single-turn prompts (recommended)

You can omit `<sound>` entirely and let the processor prepend the expanded tokens:

```python
prompt = "Transcribe the input speech."
inputs = processor(prompt, audio_array, padding_side="left", tensor_type="pt")
```

### Explicit placeholder control (advanced)

If you include `<sound>` in your prompt, the number of placeholders must equal the number of windows derived from `chunk_length` and `sampling_rate`:

```python
# For a short clip that fits in a single window:
prompt = "<sound>\nDescribe the audio in detail."
inputs = processor(prompt, audio_array, padding_side="left", tensor_type="pt")
```

Notes:

* The processor computes window counts automatically. If your audio spans multiple windows, include the same number of `<sound>` placeholders, in order.
* When placeholders are present, each is expanded to the correct number of post-pool frame tokens internally.

### Batch inference

```python
texts = [
    "Transcribe the input speech.",
    "Describe the ambience and any notable sound events."
]
audios = [audio1, audio2]
inputs = processor(texts, audios, padding_side="left", tensor_type="pt")
generate_ids = model.generate(**inputs, max_new_tokens=512)
answers = processor.batch_decode(generate_ids, skip_special_tokens=True)
answers = [a.split("\nassistant\n")[-1] for a in answers]
```

## Long audio and windowing

* The feature extractor’s `chunk_length` and `sampling_rate` determine window size.
  The default setup processes approximately 30-second windows at 16 kHz mono.
* The processor caps the total number of windows per sample to a practical limit (about 10 minutes by default).
* For each window:

  * `L_mel` is the padded mel length.
  * A conv stack reduces time as `L1 = (L_mel - 1) // 2 + 1`.
  * Post-pool frames are `K = (L1 - 2) // 2 + 1`.
  * The processor expands `<sound>` exactly `K` times for that window.

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
  Fix: Let the processor handle expansion. If you include `<sound>` explicitly, ensure the number of placeholders equals the number of windows for that sample.

* Error: “Sample X: found N placeholders but audio was split into M window(s).”
  Cause: Mismatch between manual placeholders and automatic windowing.
  Fix: Adjust the number of placeholders or omit them entirely.

* Empty or truncated outputs when batching
  Use left padding for batched generation and remove the prompt prefix by splitting on `"\nassistant\n"` as shown in the quickstart.


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

<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-28.*


# Apertus 1.5

> [!WARNING]
> Both bundled tokenizers must run in `float32`: their code assignment is an argmax over codebook scores, and
> half precision flips a significant fraction of codes (~8% for the vision tokenizer in bf16). They are kept in
> `float32` automatically when the model is loaded with `dtype=torch.float16`/`bfloat16`
> (`_keep_in_fp32_modules_strict`). The keep applies to `from_pretrained` only: manually casting the loaded
> model (`.half()`, `.to(dtype)`) or running the tokenizers under `torch.autocast` re-introduces the flips.

> [!WARNING]
> `Apertus1p5VisionTokenizerModel` is an inference-only vision-tokenizer port: it implements only inference-time codebook
> scoring and hard `argmax` indices, and omits IBQ's differentiable training path, tokenizer losses, and decoder.
> Calling `.train()` does not restore the omitted components, and `encode` (which respects the caller's gradient
> mode, like all public Transformers methods) returns indices that stop gradients. The Apertus language backbone
> retains the standard differentiable forward and loss APIs.

## Overview

Apertus 1.5 is a multimodal model (image + audio + text → text) by the
[Swiss AI Initiative](https://huggingface.co/swiss-ai) that extends the [Apertus](./apertus) language model
([Apertus: Democratizing Open and Compliant LLMs for Global Language Environments](https://huggingface.co/papers/2509.14233)) by continued pretraining
with discrete-token early fusion: frozen tokenizers turn images and audio into discrete codes that are mapped
into an enlarged text vocabulary by fixed offsets, so all modalities share the backbone's embedding table and
are modeled as a single token stream.

The model composes three parts:

- an **Apertus 1.5 language backbone** (`Apertus1p5TextModel`) with an enlarged input vocabulary covering the
  text tokens plus 131,072 visual and 4,096 audio codes. Its **output layer is pruned**: checkpoints keep only
  the text-token rows of the LM head (`output_vocab_size` in the config, 131,072 for the released
  checkpoints), so the model can embed multimodal tokens as inputs but can only ever generate text ids,
- **`Apertus1p5VisionTokenizerModel`**, an encode-only port of the
  [EMU3.5 Vision Tokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer) by BAAI
  ([Emu3.5: Native Multimodal Models are World Learners](https://huggingface.co/papers/2510.26583), with
  [IBQ](https://huggingface.co/papers/2412.02692) quantization): 16× spatial downsampling, one code per 16×16
  patch,
- **[WavTokenizer](./wavtokenizer)** ([paper](https://huggingface.co/papers/2408.16532)) as the audio codec:
  40 codes per second of 24 kHz mono audio.

> [!NOTE]
> Consequences of the pruned output layer: logits returned without `labels` are padded to the full vocabulary
> width, with `torch.finfo(dtype).min` scores for the input-only multimodal tail, so unconstrained generation
> (sampling, beam search, classifier-free guidance, ...) works generically and never selects a multimodal id;
> loss-only calls with `labels` return logits of the physical head width instead. Generation constraints that
> target input-only ids (`prefix_allowed_tokens_fn`, `force_words_ids`, forced tokens) are unsupported and
> silently emit ids the head has no learned distribution for. DoLa decoding (`dola_layers`) is also unsupported:
> it applies the physical LM head directly to intermediate hidden states, whose logits do not have the padded
> logical vocabulary width. Label positions holding input-only ids must be masked with `-100` (the model raises
> an explicit error otherwise), and `Trainer`'s `label_smoothing_factor` is unsupported (its loss diverges over
> the tail). Resizing token embeddings and tying the head to the input embeddings are rejected for pruned
> checkpoints.

Images are always encoded one at a time, even in batched inputs, because the vision tokenizer contains global
attention, so batch padding would change the codes. Same for the audio tokenizer and audio samples.
Each image contributes `(height / 16) · (width / 16)` codes
of its *resized* dimensions (the processor resizes to multiples of 16 within a pixel-area budget), and each
audio clip contributes `ceil(samples / 600)` codes.

This model was contributed by the [Swiss AI Initiative](https://huggingface.co/swiss-ai).

## Usage examples

Multimodal chat with the instruction-tuned model: the processor renders the chat template, loads and
resamples the referenced media, and expands the placeholders into the model's token stream:

```python
import torch
from transformers import Apertus1p5ForConditionalGeneration, AutoProcessor

model = Apertus1p5ForConditionalGeneration.from_pretrained(
    "swiss-ai/Apertus-v1.5-8B", dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("swiss-ai/Apertus-v1.5-8B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "What do you see in this image?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0])
```

For the base model (or full control over the prompt), call the processor directly with rendered text
containing one `<|image|>` / `<|audio|>` placeholder per media item. Media entries may be loaded objects
(PIL images, numpy waveforms) or URL / local-path strings: the processor fetches files itself and resamples
fetched audio to 24 kHz (bare waveform arrays are assumed to already be 24 kHz mono). Flat lists are
consumed left-to-right by placeholder order; nested lists (one sub-list per batch sample) give explicit
per-sample ownership with arbitrary counts (in this case the numbers of media in each list must mathc the placeholder number):

```python
# `model` and `processor` as in the quick start above; batched generation requires left padding
# (the shipped tokenizer defaults to `padding_side="left"`)
import numpy as np
from transformers.image_utils import load_image

image_a = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
image_b = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
waveform_24khz = np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0).astype(np.float32)  # 1 s of 24 kHz audio

inputs = processor(
    text=["<|image|> vs <|image|>: which image is sharper?", "Transcribe: <|audio|>"],
    images=[[image_a, image_b], []],
    audio=[[], [waveform_24khz]],
    padding=True,
    return_tensors="pt",
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

### Input expectations and validation

- **Images** are expected unscaled: pass PIL images or uint8-range pixel arrays. Per the standard
  `do_rescale` convention, float images already scaled to `[0, 1]` would be rescaled again. The image
  processor converts to RGB, resizes to multiples of 16 within the `[256², 1400²]` pixel-area budget, and
  normalizes to `[-1, 1]`, so the model always receives sizes it can tokenize.
- **The image token budget is controlled by `min_pixels` / `max_pixels`**: one token per 16×16 patch of
  the resized image means `max_pixels` caps the tokens an image can contribute (the default `1400²`
  allows up to ~7,700). Lower it to trade visual detail for shorter sequences and cheaper prefill, e.g.
  `max_pixels=512 * 512` for at most 1,024 tokens per image — either persistently on the image processor
  (`processor.image_processor.max_pixels = 512 * 512`) or per call:

  ```python
  processor(text=..., images=..., images_kwargs={"max_pixels": 512 * 512})
  # or through the chat template:
  processor.apply_chat_template(messages, processor_kwargs={"images_kwargs": {"max_pixels": 512 * 512}}, ...)
  ```
- **Audio** accepts raw numpy arrays, file paths, and URLs. A raw array is validated where possible:
  stereo or empty clips are rejected with a `ValueError`, any dtype is converted to float32, and the
  absolute scale does not matter because every clip is peak-normalized to -3 dBFS before encoding. The
  one thing that cannot be checked is the actual sample rate: a bare array carries no rate, so it is
  trusted to be 24 kHz mono, and audio recorded at another rate is accepted silently and simply
  tokenizes wrong (time-stretched). To make a rate mismatch fail loudly instead, declare it: passing
  `sampling_rate` with any value other than 24000 raises a `ValueError`. Only file and URL inputs go
  through the audio loader and are resampled to 24 kHz automatically.
- **Placeholder counts** are validated strictly in both directions: per sample for nested media lists, as
  totals for flat lists, with a `ValueError` on any mismatch.

## Apertus1p5Config

[[autodoc]] Apertus1p5Config

## Apertus1p5TextConfig

[[autodoc]] Apertus1p5TextConfig

## Apertus1p5VisionTokenizerConfig

[[autodoc]] Apertus1p5VisionTokenizerConfig

## Apertus1p5Processor

[[autodoc]] Apertus1p5Processor
    - __call__

## Apertus1p5ImageProcessor

[[autodoc]] Apertus1p5ImageProcessor
    - preprocess

## Apertus1p5VisionTokenizerModel

[[autodoc]] Apertus1p5VisionTokenizerModel
    - encode

## Apertus1p5TextModel

[[autodoc]] Apertus1p5TextModel
    - forward

## Apertus1p5TextForCausalLM

[[autodoc]] Apertus1p5TextForCausalLM
    - forward

## Apertus1p5Model

[[autodoc]] Apertus1p5Model
    - forward

## Apertus1p5ForConditionalGeneration

[[autodoc]] Apertus1p5ForConditionalGeneration
    - forward

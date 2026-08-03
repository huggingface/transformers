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
*This model was contributed to Hugging Face Transformers on 2026-06-03.*


# Gemma4 Unified

## Overview

Gemma 4 12B Unified is an **encoder-free** multimodal model with pretrained and instruction-tuned variants. Unlike [standard Gemma 4](./gemma4), which uses dedicated encoder towers, Gemma 4 12B Unified projects raw inputs directly into the language model's embedding space through lightweight linear pipelines. This results in a simpler architecture while maintaining strong multimodal performance.

Key differences from standard Gemma 4:
- **No Vision Tower**: Raw pixel patches are projected directly into LM space via a `Dense + LayerNorm` pipeline with factorized 2D positional embeddings, replacing the vision encoder.
- **No Audio Tower**: Raw 16 kHz waveform samples are chunked into fixed-length frames and projected through a simple `RMSNorm → Linear` pipeline, replacing the mel spectrogram + Conformer encoder.
- **Shared Multimodal Pipeline**: Both vision and audio use the same `Gemma4UnifiedMultimodalEmbedder` (RMSNorm → Linear) for the final projection to text hidden space.

You can find the original Gemma 4 12B Unified checkpoints under the [Gemma 4](https://huggingface.co/collections/google/gemma-4) release.

### Encoder-Free Vision Pipeline

The key architectural difference from standard Gemma 4 is the removal of the vision encoder tower. Instead, Gemma 4 12B Unified processes images through a lightweight pipeline:

1. **Patchification**: Images are split into `16×16` pixel patches
2. **Patch Merging**: Adjacent `3×3` patches are merged into `48×48` model patches, each with `48² × 3 = 6,912` raw pixel channels
3. **Projection**: `LayerNorm → Dense → LayerNorm` projects each merged patch into the LM embedding dimension
4. **Positional Embedding**: Factorized 2D positional embeddings are added (separate learned embeddings for x and y axes, summed together)
5. **Final Norm**: A final `LayerNorm` is applied
6. **Multimodal Embedder**: `RMSNorm → Linear` projects to the text hidden size

Like standard Gemma 4, the model processes **images of different sizes** using a **fixed-budget number of tokens**. The same constraints apply:
- The total number of pixels must fit within a patch budget
- Both height and width must be divisible by **48** (= patch size 16 × pooling kernel 3)

> [!IMPORTANT]
> Gemma 4 12B Unified does **not** apply mean/std normalization. The model's own patch embedding layer handles the final scaling internally.

The number of soft tokens per image is configurable. The supported options and default (**280 soft tokens**) are:

| Soft Tokens | Patches (before pooling) | Approx. Image Area |
|:-----------:|:------------------------:|:-------------------:|
| 70          | 630                      | ~161K pixels        |
| 140         | 1,260                    | ~323K pixels        |
| **280**     | **2,520**                | **~645K pixels**    |
| 560         | 5,040                    | ~1.3M pixels        |
| 1,120       | 10,080                   | ~2.6M pixels        |

### Encoder-Free Audio Pipeline

The audio pipeline is similarly simplified. Instead of computing mel spectrograms and processing them through a Conformer encoder, raw 16 kHz waveform samples are:

1. **Chunked** into fixed-length frames of 640 samples each (40ms per frame at 16 kHz)
2. **Projected** directly through `RMSNorm → Linear` via the shared `Gemma4UnifiedMultimodalEmbedder`

Since there is **no downsampling**, the number of output soft tokens equals the number of input frames: `ceil(num_samples / 640)`.

## Usage examples

The example below demonstrates how to generate text based on an image and an audio sample with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


pipe = pipeline(
    task="any-to-any",
    model="google/gemma-4-12B-it",
)

image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type": "text",
                "text": "What is shown in this image?"
            }
        ]
    }
]

image_output = pipe(image_messages, return_full_text=False)
print(image_output[0]["generated_text"])

audio_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please transcribe the following audio:"},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
            },
        ],
    }
]

audio_output = pipe(audio_messages, return_full_text=False)
print(audio_output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

### Image

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model = AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-12B-it",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-12B-it"
)

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
```

### Audio

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please transcribe the following audio:"},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
            },
        ],
    }
]

model = AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-12B-it",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-12B-it"
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)

input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

</hfoption>
</hfoptions>

## Gemma4UnifiedAudioConfig

[[autodoc]] Gemma4UnifiedAudioConfig

## Gemma4UnifiedConfig

[[autodoc]] Gemma4UnifiedConfig

## Gemma4UnifiedTextConfig

[[autodoc]] Gemma4UnifiedTextConfig

## Gemma4UnifiedVisionConfig

[[autodoc]] Gemma4UnifiedVisionConfig

## Gemma4UnifiedAudioFeatureExtractor

[[autodoc]] Gemma4UnifiedAudioFeatureExtractor
    - __call__

## Gemma4UnifiedImageProcessor

[[autodoc]] Gemma4UnifiedImageProcessor

## Gemma4UnifiedVideoProcessor

[[autodoc]] Gemma4UnifiedVideoProcessor

## Gemma4UnifiedProcessor

[[autodoc]] Gemma4UnifiedProcessor
    - __call__

## Gemma4UnifiedPreTrainedModel

[[autodoc]] Gemma4UnifiedPreTrainedModel
    - forward

## Gemma4UnifiedModel

[[autodoc]] Gemma4UnifiedModel
    - forward

## Gemma4UnifiedTextModel

[[autodoc]] Gemma4UnifiedTextModel
    - forward

## Gemma4UnifiedForCausalLM

[[autodoc]] Gemma4UnifiedForCausalLM

## Gemma4UnifiedForConditionalGeneration

[[autodoc]] Gemma4UnifiedForConditionalGeneration
    - forward

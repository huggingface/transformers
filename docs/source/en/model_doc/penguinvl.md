<!--Copyright 2025 Tencent and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PenguinVL

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**Penguin-VL** is a compact vision-language model family built to study how far multimodal efficiency can be pushed by redesigning the **vision encoder**, rather than only scaling data or model size.

Most modern VLMs rely on vision encoders pretrained with large-scale **contrastive objectives** such as CLIP or SigLIP. Penguin-VL argues that this setup can be suboptimal for multimodal reasoning because contrastive learning favors coarse category-level invariances over the fine-grained signals needed for **OCR, document understanding, dense captioning, and complex reasoning**. Instead, Penguin-VL introduces **Penguin-Encoder**, a vision encoder **initialized from a text-only LLM**, so the visual backbone starts closer to the language model representation space and learns more data-efficiently.

<img src="https://github.com/tencent-ailab/Penguin-VL/blob/master/assets/framework.png?raw=true"
alt="drawing" width="600"/>
<small> PenguinVL architecture. Details are in <a href="https://huggingface.co/papers/2603.06569">technical report.</a> </small>

This model was contributed by [Cyril666](https://huggingface.co/Cyril666).

## Usage example

### Single media inference

PenguinVL accepts both images and videos as input. Use `processor.process_vision_info` to extract visual inputs from messages **before** calling `apply_chat_template`.

```python
import torch
from transformers import PenguinVLProcessor, PenguinVLForConditionalGeneration

model = PenguinVLForConditionalGeneration.from_pretrained(
    "tencent/Penguin-VL-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = PenguinVLProcessor.from_pretrained("tencent/Penguin-VL-8B")

# Image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

images, frame_types = processor.process_vision_info(messages)
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    images=images,
    text=text,
    frame_types=frame_types,
    return_tensors="pt",
).to(model.device)

inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Video inference

```python
import torch
from transformers import PenguinVLProcessor, PenguinVLForConditionalGeneration

model = PenguinVLForConditionalGeneration.from_pretrained(
    "tencent/Penguin-VL-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = PenguinVLProcessor.from_pretrained("tencent/Penguin-VL-8B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

# process_vision_info must be called before apply_chat_template for videos
# It samples frames at `fps`, caps at `max_frames`, and annotates timestamps
images, frame_types = processor.process_vision_info(messages, fps=1, max_frames=128)
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    images=images,
    text=text,
    frame_types=frame_types,
    return_tensors="pt",
).to(model.device)

inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Batch mixed media inference

The model can batch inputs composed of mixed samples (images, videos, and text).

```python
import torch
from transformers import PenguinVLProcessor, PenguinVLForConditionalGeneration

model = PenguinVLForConditionalGeneration.from_pretrained(
    "tencent/Penguin-VL-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = PenguinVLProcessor.from_pretrained("tencent/Penguin-VL-8B")

conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "Summarize this video."},
        ],
    }
]

conversation3 = [
    {
        "role": "user",
        "content": "What is the capital of France?",
    }
]

all_images = []
all_frame_types = []
all_texts = []
for conv in [conversation1, conversation2, conversation3]:
    imgs, fts = processor.process_vision_info(conv, fps=1, max_frames=64)
    if imgs is not None:
        all_images.extend(imgs)
    if fts is not None:
        all_frame_types.extend(fts)
    all_texts.append(processor.apply_chat_template(conv, add_generation_prompt=True))

inputs = processor(
    images=all_images if all_images else None,
    text=all_texts,
    frame_types=all_frame_types if all_frame_types else None,
    padding=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```
### process_vision_info function

`process_vision_info` extracts and loads visual inputs (images and video frames) from Qwen2-VL style conversation messages. It walks through the messages, collects images/video frames in order, and for video clips samples frames at the given `fps` (capped at `max_frames`). Video content items in `messages` are enriched in-place with `num_frames` and `timestamps` so that `apply_chat_template` can emit per-frame timestamp prefixes.

**Important:** You must call `process_vision_info` **before** `apply_chat_template`, because it modifies the `messages` in-place when processing videos.

Supported content block formats:

**Image** — URL (HTTP or file) or PIL Image:

```python
{"type": "image", "image": "https://example.com/photo.jpg"}
{"type": "image", "image": "file:///path/to/image.png"}
{"type": "image", "image": <PIL.Image.Image>}
```

**Video** — URL, or list of frames with timestamps:

```python
{"type": "video", "video": "https://example.com/clip.mp4"}
{"type": "video", "video": ["file:///path/frame1.jpg", ...], "timestamps": [0, ...]}
{"type": "video", "video": [<PIL.Image.Image>, ...], "timestamps": [0, ...]}
```

### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with Flash Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using Flash Attention-2, simply add `attn_implementation="flash_attention_2"` when loading the model:

```python
import torch
from transformers import PenguinVLForConditionalGeneration

model = PenguinVLForConditionalGeneration.from_pretrained(
    "tencent/Penguin-VL-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

## Notes

- Use `min_pixels` and `max_pixels` to control image resolution and memory usage.

    ```python
    from transformers import PenguinVLProcessor

    processor = PenguinVLProcessor.from_pretrained(
        "tencent/Penguin-VL-8B",
        min_pixels=256 * 14 * 14,
        max_pixels=1280 * 14 * 14,
    )
    ```

- For video inputs, `process_vision_info` must be called **before** `apply_chat_template`. It samples frames at the given `fps`, caps total frames at `max_frames`, and annotates each video entry in `messages` with `num_frames` and `timestamps` so the chat template can emit per-frame timestamp prefixes.

- Video frames are automatically classified as **keyframes (K)** or **intermediate frames (I)** via the TRA mechanism. Keyframes receive a smaller spatial merge factor (better quality) and intermediate frames receive a larger one (higher compression). This is handled automatically when you pass `frame_types` to the processor.

- Pass `frame_types=None` (or omit it) if you are processing only images.

## PenguinVLConfig

[[autodoc]] PenguinVLConfig

## PenguinVLVisionConfig

[[autodoc]] PenguinVLVisionConfig

## PenguinVLImageProcessor

[[autodoc]] PenguinVLImageProcessor
    - preprocess

## PenguinVLImageProcessorFast

[[autodoc]] PenguinVLImageProcessorFast
    - preprocess

## PenguinVLProcessor

[[autodoc]] PenguinVLProcessor
    - __call__

## PenguinVLVisionModel

[[autodoc]] PenguinVLVisionModel
    - forward

## PenguinVLModel

[[autodoc]] PenguinVLModel
    - forward
    - get_image_features

## PenguinVLForConditionalGeneration

[[autodoc]] PenguinVLForConditionalGeneration
    - forward
    - get_image_features

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
*This model was published in HF papers on 2025-08-16 and contributed to Hugging Face Transformers on 2026-08-06.*

# Ovis2.5

## Overview

[Ovis2.5](https://huggingface.co/papers/2508.11737) is a multimodal model from
[AIDC-AI](https://github.com/AIDC-AI/Ovis) for image, video, and text understanding. It combines a native-resolution
vision encoder with a visual tokenizer and a Qwen3 language backbone. Unlike a fixed-resolution tiled encoder, the
vision encoder keeps variable-resolution visual inputs as a sequence of patches. Ovis2.5 also provides an optional
thinking prompt format for tasks that benefit from a longer reasoning response.

Two checkpoints are available:

- [AIDC-AI/Ovis2.5-2B](https://huggingface.co/AIDC-AI/Ovis2.5-2B)
- [AIDC-AI/Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B)

> [!IMPORTANT]
> The original checkpoint repositories were exported for a custom `trust_remote_code` implementation. The native
> `Ovis2_5*` classes require a checkpoint revision or local directory with native configuration, tokenizer special
> tokens, and processor metadata. Converting model weight names alone does not convert those supporting files. The
> examples below assume that the selected checkpoint has been converted to the native Transformers format.

Convert either released checkpoint without executing its remote modeling code:

```bash
python src/transformers/models/ovis2_5/convert_ovis2_5_weights_to_hf.py \
    --input_model_id AIDC-AI/Ovis2.5-2B \
    --output_dir Ovis2.5-2B-hf

python src/transformers/models/ovis2_5/convert_ovis2_5_weights_to_hf.py \
    --input_model_id AIDC-AI/Ovis2.5-9B \
    --output_dir Ovis2.5-9B-hf
```

The conversion requires the Transformers vision dependencies, including Torchvision. It applies the registered native
weight mapping, fails on missing, unexpected, or mismatched weights, registers the five native visual special tokens
without resizing the text embedding matrix, writes native image and video processor metadata, and reloads the converted
checkpoint.

## Image inference

Use [`Ovis2_5Processor.apply_chat_template`] to load the image, format the conversation, and tokenize it in one call.
Replace the 2B directory with the converted 9B directory to use the larger model.

```python
from transformers import Ovis2_5ForConditionalGeneration, Ovis2_5Processor


model_id = "./Ovis2.5-2B-hf"  # or "./Ovis2.5-9B-hf"
processor = Ovis2_5Processor.from_pretrained(model_id)
model = Ovis2_5ForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

Multiple images can be supplied by adding more `{"type": "image", ...}` items to the message in the same order in which
they should appear in the prompt.

## Video inference

The official example uniformly samples eight frames. Load and sample the video before placing the decoded frames in a
`video` content item. The processor returns `pixel_values_videos` and `video_grid_thw` along with the tokenized prompt.
Ovis2.5 accepts exactly one video per request and does not support mixing images and video in the same request.

```python
from transformers.video_utils import load_video


video_frames, _ = load_video("path/to/video.mp4", num_frames=8)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_frames},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
    processor_kwargs={"videos_kwargs": {"max_pixels": 896 * 896}},
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Thinking mode

Set `enable_thinking=True` when applying the chat template to leave the model's reasoning block open. Set it to `False`
to request the immediate-answer prompt format.

```python
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=True,
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
```

`enable_thinking` changes the chat template; it is not a generation argument and does not impose a token budget. The
two-stage `thinking_budget` behavior implemented by the original custom remote code is not part of the native
Transformers generation API.

## Ovis2_5Config

[[autodoc]] Ovis2_5Config

## Ovis2_5VisionConfig

[[autodoc]] Ovis2_5VisionConfig

## Ovis2_5Processor

[[autodoc]] Ovis2_5Processor
    - __call__

## Ovis2_5ImageProcessor

[[autodoc]] Ovis2_5ImageProcessor
    - preprocess

## Ovis2_5ImageProcessorPil

[[autodoc]] Ovis2_5ImageProcessorPil
    - preprocess

## Ovis2_5VideoProcessor

[[autodoc]] Ovis2_5VideoProcessor
    - preprocess

## Ovis2_5VisionModel

[[autodoc]] Ovis2_5VisionModel
    - forward

## Ovis2_5Model

[[autodoc]] Ovis2_5Model
    - forward
    - get_image_features
    - get_video_features

## Ovis2_5ForConditionalGeneration

[[autodoc]] Ovis2_5ForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features

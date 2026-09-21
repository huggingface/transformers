<!--Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-09-16 and contributed to Hugging Face Transformers on 2026-09-21.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# MiniCPM-V 4.7

[MiniCPM-V](https://huggingface.co/papers/2509.18154) is a series of efficient multimodal large language models developed by [OpenBMB](https://github.com/OpenBMB). Like [MiniCPM-V 4.6](./minicpmv4_6.md), the MiniCPM-V 4.7 architecture pairs a [SigLIP](./siglip.md) vision encoder that has a window-attention merger with a [Qwen3.5](./qwen3_5.md) language model backbone, and supports both 4x and 16x visual downsampling modes.

The main addition over 4.6 is *canvas M-RoPE*: instead of numbering visual tokens along a single 1-D sequence, the model lays every image out on a 2-D canvas and assigns each visual token a `(temporal, height, width)` position, so slices of the same image keep their spatial relationship and video frames keep their temporal order.

This model was contributed by [OpenBMB](https://huggingface.co/openbmb).
The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

> [!NOTE]
> Passing `use_image_id` to a processor will number several images in one prompt so the text can refer to them individually. It applies to images only: a video is a single temporal sequence of frames rather than several addressable visuals, which is how the model was trained, so the setting is ignored for video inputs.

## Usage example

### Inference with Pipeline

```python
from transformers import pipeline

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

pipe = pipeline("image-text-to-text", model="openbmb/MiniCPM-V-4_7")
outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
```

### Inference on a single image

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_checkpoint = "openbmb/MiniCPM-V-4_7"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=100)
decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_output)
```

### Downsampling mode

MiniCPM-V 4.7 supports two visual downsampling modes:

- **16x** (default): More aggressive downsampling, fewer visual tokens, faster inference.
- **4x**: Less downsampling, more visual tokens, better for detail-rich tasks.

You can change the downsampling mode at runtime by passing `downsample_mode` via `processor_kwargs` and to `model.generate`:

```python
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    processor_kwargs={"downsample_mode": "4x"},
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=100, downsample_mode="4x")
```

### Thinking mode

The model supports a thinking mode controlled by `enable_thinking` in the chat template. When enabled, the model generates internal reasoning before providing the final answer:

```python
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    enable_thinking=True,
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=1024)
```

To disable thinking (default for evaluation):

```python
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    enable_thinking=False,
).to(model.device, dtype=model.dtype)
```

### Video inference

MiniCPM-V 4.7 supports video understanding.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/video.mp4"},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=200)
decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_output)
```

## MiniCPMV4_7Config

[[autodoc]] MiniCPMV4_7Config

## MiniCPMV4_7VisionConfig

[[autodoc]] MiniCPMV4_7VisionConfig

## MiniCPMV4_7VisionPreTrainedModel.

[[autodoc]] MiniCPMV4_7VisionPreTrainedModel
    - forward

## MiniCPMV4_7VisionModel

[[autodoc]] MiniCPMV4_7VisionModel
    - forward

## MiniCPMV4_7Model

[[autodoc]] MiniCPMV4_7Model
    - forward
    - get_image_features
    - get_video_features

## MiniCPMV4_7ForConditionalGeneration

[[autodoc]] MiniCPMV4_7ForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features

## MiniCPMV4_7Processor

[[autodoc]] MiniCPMV4_7Processor
    - __call__

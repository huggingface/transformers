<!--Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-09-16 and added to Hugging Face Transformers on 2026-04-28.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# MiniCPM-V

[MiniCPM-V](https://huggingface.co/papers/2509.18154) is a series of efficient multimodal large language models developed by [OpenBMB](https://github.com/OpenBMB). The MiniCPM-V 4.6 architecture uses a [SigLIP](siglip) vision encoder with a window-attention merger and a [Qwen3.5](qwen3_5) language model backbone, supporting both 4x and 16x visual downsampling modes.

This model was contributed by [OpenBMB](https://huggingface.co/openbmb).
The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

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

pipe = pipeline("image-text-to-text", model="openbmb/MiniCPM-V-4_6")
outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
```

### Inference on a single image

> [!NOTE]
> The model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_checkpoint = "openbmb/MiniCPM-V-4_6"
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

MiniCPM-V 4.6 supports two visual downsampling modes:

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

### Image processing backend

MiniCPM-V 4.6 provides two image processing backends:

- **torchvision** (default): Uses `torchvision.transforms` for image resizing.
- **pil**: Uses `PIL.Image.resize`, matching the original implementation.

To use the PIL backend:

```python
from transformers import AutoProcessor, AutoImageProcessor

processor = AutoProcessor.from_pretrained(model_checkpoint)
processor.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, backend="pil")
```

### Video inference

MiniCPM-V 4.6 supports video understanding.

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

If you already have the rendered prompt string, you can call `processor(text=..., videos=[...])` directly instead.

## MiniCPMV4_6Config

[[autodoc]] MiniCPMV4_6Config

## MiniCPMV4_6VisionConfig

[[autodoc]] MiniCPMV4_6VisionConfig

## MiniCPMV4_6Model

[[autodoc]] MiniCPMV4_6Model
    - forward
    - get_image_features

## MiniCPMV4_6ForConditionalGeneration

[[autodoc]] MiniCPMV4_6ForConditionalGeneration
    - forward
    - get_image_features

## MiniCPMV4_6Processor

[[autodoc]] MiniCPMV4_6Processor
    - __call__

## MiniCPMV4_6ImageProcessor

[[autodoc]] MiniCPMV4_6ImageProcessor
    - preprocess

## MiniCPMV4_6ImageProcessorPil

[[autodoc]] MiniCPMV4_6ImageProcessorPil
    - preprocess

## MiniCPMV4_6VideoProcessor

[[autodoc]] MiniCPMV4_6VideoProcessor
    - preprocess

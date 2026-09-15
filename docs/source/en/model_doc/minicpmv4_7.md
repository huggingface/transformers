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
*This model was published in HF papers on 2026--xx-xx and contributed to Hugging Face Transformers on 2026-xx-xx.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# MiniCPM-V 4.7

[MiniCPM-V](https://huggingface.co/papers/2509.18154) is a series of efficient multimodal large language models developed by [OpenBMB](https://github.com/OpenBMB). Like [MiniCPM-V 4.6](minicpmv4_6), the MiniCPM-V 4.7 architecture pairs a [SigLIP](siglip) vision encoder that has a window-attention merger with a [Qwen3.5](qwen3_5) language model backbone, and supports both 4x and 16x visual downsampling modes.

The main addition over 4.6 is *canvas M-RoPE*: instead of numbering visual tokens along a single 1-D sequence, the model lays every image out on a 2-D canvas and assigns each visual token a `(temporal, height, width)` position, so slices of the same image keep their spatial relationship and video frames keep their temporal order. See [Canvas M-RoPE](#canvas-m-rope) for the configuration this requires.

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

pipe = pipeline("image-text-to-text", model="openbmb/MiniCPM-V-4_7")
outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
```

### Inference on a single image

> [!NOTE]
> The model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.

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

### Canvas M-RoPE

The processor returns a `target_sizes_mrope` entry next to `input_ids` and `pixel_values`. It holds the patch grid of every visual input, in the order the placeholders appear in the prompt, and the model turns it into 3-D positions during `forward`. Both `apply_chat_template(..., tokenize=True, return_dict=True)` and a direct `processor(text=..., images=...)` call produce it, so the default path needs no extra work.

The layout is recovered by scanning `input_ids` for the structural markers around each visual span, so the checkpoint config must carry their token ids:

```python
model.config.image_start_id, model.config.image_end_id
model.config.slice_start_id, model.config.slice_end_id
model.config.newline_id
```

The conversion script resolves these from the tokenizer and stores them in `config.json`, so released checkpoints already carry them. If a checkpoint ships without them, the model raises instead of quietly falling back to 1-D positions, because that fallback degrades quality on every image and video input.

> [!TIP]
> Canvas positions are built from `input_ids`. When you generate from `inputs_embeds` only, pass `input_ids` for the first forward pass as well, otherwise the model warns and falls back to 1-D positions.

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

### Image processing backend

MiniCPM-V 4.7 provides two image processing backends:

- **torchvision** (default): Uses `torchvision.transforms` for image resizing.
- **pil**: Uses `PIL.Image.resize`, matching the original implementation.

To use the PIL backend:

```python
from transformers import AutoProcessor, AutoImageProcessor

processor = AutoProcessor.from_pretrained(model_checkpoint)
processor.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, backend="pil")
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

If you already have the rendered prompt string, you can call `processor(text=..., videos=[...])` directly instead.

> [!NOTE]
> `use_image_id` numbers several images in one prompt so the text can refer to them individually. It applies to images only: a video is a single temporal sequence of frames rather than several addressable visuals, which is how the model was trained, so the setting is ignored for video inputs.

## MiniCPMV4_7Config

[[autodoc]] MiniCPMV4_7Config

## MiniCPMV4_7VisionConfig

[[autodoc]] MiniCPMV4_7VisionConfig

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

## MiniCPMV4_7ImageProcessor

[[autodoc]] MiniCPMV4_7ImageProcessor
    - preprocess

## MiniCPMV4_7ImageProcessorPil

[[autodoc]] MiniCPMV4_7ImageProcessorPil
    - preprocess

## MiniCPMV4_7VideoProcessor

[[autodoc]] MiniCPMV4_7VideoProcessor
    - preprocess

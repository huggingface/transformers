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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# MiniCPM-OCR

MiniCPM-OCR is a document-understanding model from the [MiniCPM-V](https://huggingface.co/papers/2509.18154) family developed by [OpenBMB](https://github.com/OpenBMB). It keeps the [MiniCPM-V 4.6](./minicpmv4_6.md) architecture -- a [SigLIP](./siglip.md) vision encoder with a window-attention merger in front of a [Qwen3.5](./qwen3_5.md) language model -- and specializes it for reading text-dense images such as scanned pages, screenshots, receipts and tables.

This model was contributed by [OpenBMB](https://huggingface.co/openbmb).
The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

> [!NOTE]
> Passing `use_image_id` to the processor will number several images in one prompt so the text can refer to them individually. It applies to images only; the setting is ignored for video inputs.

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
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/invoice.png",
            },
            {"type": "text", "text": "Transcribe all the text in this document."},
        ],
    },
]

pipe = pipeline("image-text-to-text", model="openbmb/MiniCPM-OCR-1.0")
outputs = pipe(text=messages, max_new_tokens=256, return_full_text=False)
outputs[0]["generated_text"]
```

### Inference on a single image

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_checkpoint = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/invoice.png"},
            {"type": "text", "text": "Convert this page to markdown."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512)
decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_output)
```

### Downsampling mode

MiniCPM-OCR supports both visual downsampling modes of the MiniCPM-V family:

- **4x** (default): Less downsampling, more visual tokens, which is what document and screenshot transcription needs.
- **16x**: More aggressive downsampling, fewer visual tokens, faster inference on natural images.

The mode has to be passed to the processor and to `model.generate` together, because the processor counts the visual placeholders the vision tower is expected to produce:

```python
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    processor_kwargs={"downsample_mode": "4x"},
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512, downsample_mode="4x")
```

A per-call `downsample_mode` applies to that call only and leaves the processor unchanged, so the
same processor can serve both modes concurrently. When calling the processor directly instead of
going through `apply_chat_template`, pass it as a plain keyword argument:

```python
inputs = processor(images=[image], text=[text], return_tensors="pt", downsample_mode="4x")
```

To make a mode the default instead, set it when loading the processor:

```python
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0", downsample_mode="4x")
```

### Small image resizing

Images too small to be sliced are upscaled to `scale_resolution` by default. Two options change
that, which is useful for screenshots and crops where upscaling only blurs the glyphs:

- `disable_small_image_upsize`: keep those images at their native resolution, aligned to the patch
  grid only.
- `adaptive_small_image_upsize`: decide per image instead. Small dense text is upscaled to
  `scale_resolution`, images below `adaptive_upsize_tiny_pixels` are upscaled to
  `adaptive_upsize_tiny_target_resolution`, and the rest follow `disable_small_image_upsize`.

```python
processor = AutoProcessor.from_pretrained(
    "openbmb/MiniCPM-OCR-1.0",
    disable_small_image_upsize=True,
    adaptive_small_image_upsize=True,
)
```

Glyph height and density are measured with an Otsu threshold followed by connected component
analysis, so `adaptive_small_image_upsize` requires
[OpenCV](https://pypi.org/project/opencv-python/) and raises without it, rather than silently
falling back to a different rule; `adaptive_upsize_font_px` and `adaptive_upsize_dense_per_kpx`
set the thresholds.

Like `downsample_mode`, all of these can be overridden per call as well:

```python
inputs = processor(
    images=[image],
    text=[text],
    return_tensors="pt",
    disable_small_image_upsize=True,
    adaptive_small_image_upsize=True,
)
```

### Batched inference

```python
messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/invoice.png"},
                {"type": "text", "text": "What is the total amount?"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
                {"type": "text", "text": "Is there any text in this image?"},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt", padding=True,
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=256)
decoded_outputs = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

## MiniCPMOcrConfig

[[autodoc]] MiniCPMOcrConfig

## MiniCPMOcrVisionConfig

[[autodoc]] MiniCPMOcrVisionConfig

## MiniCPMOcrVisionPreTrainedModel

[[autodoc]] MiniCPMOcrVisionPreTrainedModel
    - forward

## MiniCPMOcrVisionModel

[[autodoc]] MiniCPMOcrVisionModel
    - forward

## MiniCPMOcrModel

[[autodoc]] MiniCPMOcrModel
    - forward
    - get_image_features
    - get_video_features

## MiniCPMOcrForConditionalGeneration

[[autodoc]] MiniCPMOcrForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features

## MiniCPMOcrProcessor

[[autodoc]] MiniCPMOcrProcessor
    - __call__

## MiniCPMOcrImageProcessor

[[autodoc]] MiniCPMOcrImageProcessor
    - preprocess

## MiniCPMOcrImageProcessorPil

[[autodoc]] MiniCPMOcrImageProcessorPil
    - preprocess

## MiniCPMOcrVideoProcessor

[[autodoc]] MiniCPMOcrVideoProcessor
    - preprocess

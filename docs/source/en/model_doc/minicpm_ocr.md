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
*This model was published in HF papers on 2025-09-16 and contributed to Hugging Face Transformers on 2026-10-26.*

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


## Usage example

<hfoptions id="usage">
<hfoption id="Inference with Pipeline">

```python
from transformers import pipeline

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg",
            },
            {"type": "text", "text": "Transcribe all the text in this document."},
        ],
    },
]

pipe = pipeline("image-text-to-text", model="openbmb/MiniCPM-OCR-1.0")
outputs = pipe(text=messages, max_new_tokens=256, return_full_text=False)
print(outputs[0]["generated_text"])
# PSPS
# Document No : TD01167104
# Date : 25/12/2018 8:13:39 PM
# Cashier : MANIS
# Member :
#
# CASH BILL
#
# CODE/DESC PRICE Disc AMOUNT
# QTY RM
# 9556939040118 KF MODELLING CLAY KIDDY FISH
# 1 PC * 9.00 0.00 9.00
# Total : 9.00
# ...
```

</hfoption>

<hfoption id="Inference on a single image">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"},
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
# But, let's break this down and convert the page to markdown.
#
# ## R&D QUALITY IMPROVEMENT
# ### SUGGESTION/SOLUTION FORM
#
# #### Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez  Date: 9/3/92
#
# #### Supervisor/Manager: J. S. Wigand  R&D Group: Licensee
# ...
```

The chat template opens a thinking block unless `enable_thinking=False` is passed to
`apply_chat_template`, which is why the decoded text above starts with the model reasoning about
the page before transcribing it.

</hfoption>

<hfoption id="Downsampling mode">

MiniCPM-OCR supports both visual downsampling modes of the MiniCPM-V family:

- **4x** (default): Less downsampling, more visual tokens, which is what document and screenshot transcription needs.
- **16x**: More aggressive downsampling, fewer visual tokens, faster inference on natural images.

The mode has to be passed to the processor and to `model.generate` together, because the processor counts the visual placeholders the vision tower is expected to produce:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"},
            {"type": "text", "text": "Convert this page to markdown."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    processor_kwargs={"downsample_mode": "4x"},
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512, downsample_mode="4x")
print((inputs["input_ids"] == model.config.image_token_id).sum().item())
# 1260 visual tokens; the same page in `"16x"` mode is 315 instead
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
</hfoption>

<hfoption id="Small image resizing">

Images too small to be sliced are upscaled to `scale_resolution` by default, which is the setting
the released checkpoint was trained with. Screenshots and crops are often better kept at their
native resolution, where upscaling only blurs the glyphs, so `small_image_upsize` selects between
the two:

- `"always"` (default): upscale them to `scale_resolution`.
- `"never"`: keep them at their native resolution, aligned to the patch grid only.
- `"adaptive"`: decide per image, and per slice, from glyph size and density.

```python
from transformers import AutoProcessor, load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"
image = load_image(url).resize((224, 224))

processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read the text."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

print(processor(images=[image], text=[text], return_tensors="pt")["target_sizes"].tolist())
# [[32, 32]] -- upscaled to `scale_resolution`

processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0", small_image_upsize="never")
print(processor(images=[image], text=[text], return_tensors="pt")["target_sizes"].tolist())
# [[16, 16]] -- 224 / 14 patches per side, no upscaling
```

`"adaptive"` upscales small dense text to `scale_resolution`, images below 6272 pixels to 84, and
keeps the rest native; on a slice it never spends more tokens than the baseline tile would. Glyph
height and density are measured with an Otsu threshold followed by connected component analysis, so
it requires [OpenCV](https://pypi.org/project/opencv-python/) and raises without it, rather than
silently falling back to a different rule.

```python
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0", small_image_upsize="adaptive")
print(processor(images=[image], text=[text], return_tensors="pt")["target_sizes"].tolist())
# [[32, 32]] -- dense text detected, so this one is upscaled anyway
```

Like `downsample_mode`, it can be overridden per call as well, which keeps the 224x224 receipt at
its native resolution without reloading the processor:

```python
inputs = processor(images=[image], text=[text], return_tensors="pt", small_image_upsize="never")
print(inputs["target_sizes"].tolist())
# [[16, 16]] -- 224 / 14 patches per side, no upscaling
```

</hfoption>

<hfoption id="Batched inference">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"},
                {"type": "text", "text": "Transcribe all the text in this document."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"},
                {"type": "text", "text": "Transcribe all the text in this document."},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    processor_kwargs={"padding": True},
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=256)
decoded_outputs = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs[0])
# PSPS
# Document No: TD01167104
# Date: 25/12/2018
# Cashier: MANIS
# ...
print(decoded_outputs[1])
# ... thinking preamble ...
# **R&D QUALITY IMPROVEMENT
# SUGGESTION/SOLUTION FORM**
#
# Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez  Date: 9/3/92
# ...
```

The two pages are resized independently, so their prompts have different lengths. `padding` has to
be requested through `processor_kwargs` for `apply_chat_template` to forward it to the processor,
and the processor pads on the left so that `generate` keeps producing from the end of each prompt.

</hfoption>

</hfoptions>


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

## MiniCPMOcrVideoProcessor

[[autodoc]] MiniCPMOcrVideoProcessor
    - preprocess

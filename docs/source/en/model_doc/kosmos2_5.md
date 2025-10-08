<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

*This model was released on 2023-09-20 and added to Hugging Face Transformers on 2025-08-19.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# KOSMOS-2.5

[KOSMOS-2.5](https://huggingface.co/papers/2309.11419/) is a multimodal literate model designed for machine reading of text-intensive images. It excels in generating spatially-aware text blocks with assigned coordinates and producing structured text output in markdown format. Utilizing a shared Transformer architecture, task-specific prompts, and flexible text representations, Kosmos-2.5 performs well in end-to-end document-level text recognition and image-to-markdown text generation. The model can be adapted for various text-intensive image understanding tasks through supervised fine-tuning, making it a versatile tool for real-world applications.

<hfoptions id="usage">
<hfoption id="Kosmos2_5ForConditionalGeneration">

```py
import re
import torch
import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration,

model = Kosmos2_5ForConditionalGeneration.from_pretrained("microsoft/kosmos-2.5", dtype="auto")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")

url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<md>"
inputs = processor(text=prompt, images=image, return_tensors="pt")

height, width = inputs.pop("height"), inputs.pop("width")
raw_width, raw_height = image.size
scale_height = raw_height / height
scale_width = raw_width / width

inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])
```

</hfoption>
</hfoptions>

## Kosmos2_5Config

[[autodoc]] Kosmos2_5Config

## Kosmos2_5ImageProcessor

[[autodoc]] Kosmos2_5ImageProcessor
    - preprocess

## Kosmos2_5ImageProcessorFast

[[autodoc]] Kosmos2_5ImageProcessorFast
    - preprocess

## Kosmos2_5Processor

[[autodoc]] Kosmos2_5Processor

## Kosmos2_5Model

[[autodoc]] Kosmos2_5Model
    - forward

## Kosmos2_5ForConditionalGeneration

[[autodoc]] Kosmos2_5ForConditionalGeneration
    - forward


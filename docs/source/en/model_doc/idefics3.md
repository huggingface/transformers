<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-08-22 and added to Hugging Face Transformers on 2024-09-25 and contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [andito](https://huggingface.co/andito).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Idefics3

[Idefics3](https://huggingface.co/papers/2408.12637) is an advanced vision-language model that builds upon Idefics2 with key modifications: it incorporates Llama3 for text processing, adopts an enhanced image processing logic, and eliminates the perceiver component. The paper serves as a guide for constructing VLMs, detailing the current landscape, challenges, and future research directions. It presents the development of Idefics3-8B, which surpasses Idefics2-8B in performance, leveraging open datasets and a simple training pipeline. Additionally, the creation and release of Docmatix, a significantly larger dataset for document understanding, are highlighted.

<hfoptions id="usage">
<hfoption id="AutoModelForVision2Seq">

```py
import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", dtype="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
            {"type": "image"},
            {"type": "text", "text": "What can we see in this image?"},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "In which city is that bridge located?"},
        ]
    }
]

prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
images = [[image1, image2], [image3]]
inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts[0])
print(generated_texts[1])
```

</hfoption>
</hfoptions>

## Idefics3Config

[[autodoc]] Idefics3Config

## Idefics3VisionConfig

[[autodoc]] Idefics3VisionConfig

## Idefics3VisionTransformer

[[autodoc]] Idefics3VisionTransformer

## Idefics3Model

[[autodoc]] Idefics3Model
    - forward

## Idefics3ForConditionalGeneration

[[autodoc]] Idefics3ForConditionalGeneration
    - forward

## Idefics3ImageProcessor

[[autodoc]] Idefics3ImageProcessor
    - preprocess

## Idefics3ImageProcessorFast

[[autodoc]] Idefics3ImageProcessorFast
    - preprocess

## Idefics3Processor

[[autodoc]] Idefics3Processor
    - __call__


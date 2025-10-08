<!--Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-09-18 and added to Hugging Face Transformers on 2025-01-23.*

# Qwen2.5-VL

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2_5-vl/) is a vision-language model that advances visual recognition, object localization, document parsing, and long-video comprehension. It can accurately identify objects with bounding boxes or points and extract structured data from invoices, forms, tables, charts, and diagrams. The model handles variable-resolution images and extended videos using dynamic resolution processing and absolute time encoding, with a native dynamic-resolution Vision Transformer (ViT) and Window Attention for efficiency. Available in three sizes, Qwen2.5-VL combines strong visual reasoning and interactive capabilities with robust language performance, matching leading models like GPT-4o in tasks such as document and diagram understanding.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
pipeline = pipeline(task="image-text-to-text", model="Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe the weather in this image."},
        ]
    }
]
pipeline(text=messages,max_new_tokens=20, return_full_text=False)

```

</hfoption>
<hfoption id="Qwen2_5_VLForConditionalGeneration">

```py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type":"text",
                "text":"Describe the weather in this image.."
            }
        ]
    }

]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
</hfoptions>

## Qwen2_5_VLConfig

[[autodoc]] Qwen2_5_VLConfig

## Qwen2_5_VLProcessor

[[autodoc]] Qwen2_5_VLProcessor

## Qwen2_5_VLModel

[[autodoc]] Qwen2_5_VLModel
    - forward

## Qwen2_5_VLForConditionalGeneration

[[autodoc]] Qwen2_5_VLForConditionalGeneration
    - forward

## Qwen2_5_VLTextConfig

[[autodoc]] Qwen2_5_VLTextConfig

## Qwen2_5_VLTextModel

[[autodoc]] Qwen2_5_VLTextModel
    - forward

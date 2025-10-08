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
*This model was released on 2023-10-05 and added to Hugging Face Transformers on 2024-03-20 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# LLaVA-NeXT

[LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) introduces several key technical upgrades. Its Dynamic High Resolution or “AnyRes” mechanism enables flexible handling of various image resolutions using grid configurations like {2×2, 1×{2,3,4}, {2,3,4}×1}, improving fine-detail perception and reducing hallucinations. The Data Mixture emphasizes quality and diversity by combining curated GPT-V datasets (LAION-GPT-V, ShareGPT-4V) with a filtered 15K real-world visual instruction dataset and enhanced OCR/chart understanding data (DocVQA, SynDog-EN, ChartQA, DVQA, AI2D). Finally, scaling the LLM backbone extends support from Vicuna-1.5 (7B/13B) to larger and more flexible models like Mistral-7B and Nous-Hermes-2-Yi-34B, maintaining smooth scalability up to 34B parameters.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch  
from transformers import pipeline  

pipeline = pipeline(task="image-text-to-text", model="llava-hf/llava-v1.6-mistral-7b-hf", dtype="auto")  
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
pipeline(text=messages, max_new_tokens=20, return_full_text=False)
```

</hfoption>
<hfoption id="LlavaNextForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", dtype="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## LlavaNextConfig

[[autodoc]] LlavaNextConfig

## LlavaNextImageProcessor

[[autodoc]] LlavaNextImageProcessor
    - preprocess

## LlavaNextImageProcessorFast

[[autodoc]] LlavaNextImageProcessorFast
    - preprocess

## LlavaNextProcessor

[[autodoc]] LlavaNextProcessor

## LlavaNextForConditionalGeneration

[[autodoc]] LlavaNextForConditionalGeneration
    - forward

## LlavaNextModel

[[autodoc]] LlavaNextModel
    - forward

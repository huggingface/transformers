<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2025-04-14 and added to Hugging Face Transformers on 2025-04-18 and contributed by [yonigozlan](https://huggingface.co/yonigozlan).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# InternVL

[InternVL3](https://huggingface.co/papers/2504.10479) introduces a unified multimodal pre-training paradigm that jointly develops linguistic and visual capabilities from diverse multimodal data and text corpora. This approach resolves alignment challenges typical in post-hoc training for multimodal large language models (MLLMs). InternVL3 enhances performance and scalability through variable visual position encoding (V2PE), supervised fine-tuning (SFT), mixed preference optimization (MPO), and test-time scaling strategies. Evaluations show that InternVL3-78B achieves a top score of 72.2 on the MMMU benchmark, surpassing other open-source MLLMs and competing with leading proprietary models while maintaining strong language proficiency. The model and training data are publicly released to support further research.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="OpenGVLab/InternVL3-1B-hf", dtype="auto")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
pipeline(text=messages, max_new_tokens=50, return_full_text=False)
```

</hfoption>
<hfoption id="AutoModelForImageTextToText">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
model = AutoModelForImageTextToText.from_pretrained("OpenGVLab/InternVL3-1B-hf", dtpe="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
generate_ids = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## InternVLVisionConfig

[[autodoc]] InternVLVisionConfig

## InternVLConfig

[[autodoc]] InternVLConfig

## InternVLVisionModel

[[autodoc]] InternVLVisionModel
    - forward

## InternVLModel

[[autodoc]] InternVLModel
    - forward

## InternVLForConditionalGeneration

[[autodoc]] InternVLForConditionalGeneration
    - forward

## InternVLProcessor

[[autodoc]] InternVLProcessor

## InternVLVideoProcessor

[[autodoc]] InternVLVideoProcessor


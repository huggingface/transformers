<!--Copyright 2025 The ZhipuAI Inc. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-28 and added to Hugging Face Transformers on 2025-08-08.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Glm4vMoe

[Glm4vMoe](https://huggingface.co/papers/2508.06471) is an open-source Mixture-of-Experts (MoE) language model with 355 billion total parameters and 32 billion active per inference, designed to alternate between reasoning and direct response modes. It was trained on 23 trillion tokens using multi-stage training, expert model iteration, and reinforcement learning. The model demonstrates strong performance on reasoning and coding benchmarks—70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified—ranking third overall and second on agentic tasks despite its smaller active parameter count. A compact 106 billion-parameter version, GLM-4.5-Air, is also released to support research in reasoning and agentic AI.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="THUDM/GLM-4.1V-9B-Thinking", dtype="auto")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe this image."},
        ]
    }
]
pipeline(text=messages,max_new_tokens=20, return_full_text=False)
```

</hfoption>
<hfoption id="Glm4vMoeForConditionalGeneration">

```py
import torch
from transformers import Glm4vMoeForConditionalGeneration, AutoProcessor

model = Glm4vMoeForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking", dtype="auto")
processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
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
                "text":"Describe this image."
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

## Glm4vMoeConfig

[[autodoc]] Glm4vMoeConfig

## Glm4vMoeTextConfig

[[autodoc]] Glm4vMoeTextConfig

## Glm4vMoeTextModel

[[autodoc]] Glm4vMoeTextModel
    - forward

## Glm4vMoeModel

[[autodoc]] Glm4vMoeModel
    - forward

## Glm4vMoeForConditionalGeneration

[[autodoc]] Glm4vMoeForConditionalGeneration
    - forward

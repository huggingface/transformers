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
*This model was released on 2025-09-23 and added to Hugging Face Transformers on 2025-10-07.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Qwen3-VL

[Qwen3-V](https://huggingface.co/papers/2502.13923) is a large language model series featuring both dense and Mixture-of-Expert (MoE) architectures, with sizes ranging from 0.6 to 235 billion parameters. It introduces a unified framework combining “thinking mode” for complex reasoning and “non-thinking mode” for fast, context-driven responses, along with a thinking budget mechanism that adaptively allocates computational resources based on task complexity. Qwen3 leverages knowledge from larger flagship models to reduce resource requirements for smaller models while maintaining competitive performance, achieving state-of-the-art results in code generation, mathematical reasoning, and agent tasks. Additionally, it significantly expands multilingual support from 29 to 119 languages and is fully open-sourced under Apache 2.0 for community use.

<hfoptions id="usage">
<hfoption id="Qwen3VLForConditionalGeneration">

```py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL", dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL")

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a meterologist."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe the weather in this image."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs.pop("token_type_ids", None)

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

## Qwen3VLConfig

[[autodoc]] Qwen3VLConfig

## Qwen3VLTextConfig

[[autodoc]] Qwen3VLTextConfig

## Qwen3VLProcessor

[[autodoc]] Qwen3VLProcessor

## Qwen3VLVideoProcessor

[[autodoc]] Qwen3VLVideoProcessor

## Qwen3VLVisionModel

[[autodoc]] Qwen3VLVisionModel
    - forward

## Qwen3VLTextModel

[[autodoc]] Qwen3VLTextModel
    - forward

## Qwen3VLModel

[[autodoc]] Qwen3VLModel
    - forward

## Qwen3VLForConditionalGeneration

[[autodoc]] Qwen3VLForConditionalGeneration
    - forward

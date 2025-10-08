<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-04-17 and added to Hugging Face Transformers on 2023-12-07 and contributed by [ArthurZ](https://huggingface.co/ArthurZ) and [ybelkada](https://huggingface.co/ybelkada).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# LLaVa

[LLaVa](https://huggingface.co/papers/2304.08485) is an open-source chatbot built by fine-tuning LlamA/Vicuna on multimodal instruction-following data. It leverages a transformer architecture and incorporates a fully-connected vision-language cross-modal connector. Enhancements include using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with formatted prompts. These modifications result in state-of-the-art performance across 11 benchmarks using just 1.2M publicly available data points, completing training in about a day on a single 8-A100 node.

<hfoptions id="usage">
<hfoption id="LlavaForConditionalGeneration">

```py
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", dtype="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

generate_ids = model.generate(**inputs, max_new_tokens=30)
print(processor.batch_decode(generate_ids, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## LlavaConfig

[[autodoc]] LlavaConfig

## LlavaImageProcessor

[[autodoc]] LlavaImageProcessor
    - preprocess

## LlavaImageProcessorFast

[[autodoc]] LlavaImageProcessorFast
    - preprocess

## LlavaProcessor

[[autodoc]] LlavaProcessor

## LlavaForConditionalGeneration

[[autodoc]] LlavaForConditionalGeneration
    - forward

## LlavaModel

[[autodoc]] LlavaModel
    - forward


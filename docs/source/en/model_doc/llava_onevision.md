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
*This model was released on 2024-08-06 and added to Hugging Face Transformers on 2024-09-05 and contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# LLaVA-OneVision

[LLaVA-OneVision](https://huggingface.co/papers/2408.03326) is an open large multimodal model (LMM) designed to handle single-image, multi-image, and video tasks within a unified architecture. It advances the performance of open LMMs by enabling strong transfer learning across these visual modalities, allowing knowledge gained from one domain—such as images—to improve performance in another, like video understanding. The model’s architecture and training strategy emphasize shared visual representations that generalize effectively across different scenarios, leading to new cross-modal and cross-scenario capabilities. Overall, LLaVA-OneVision represents a consolidated, high-performing framework for multimodal perception and reasoning.

<hfoptions id="usage">
<hfoption id="LlavaOnevisionForConditionalGeneration">

```py
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## LlavaOnevisionConfig

[[autodoc]] LlavaOnevisionConfig

## LlavaOnevisionProcessor

[[autodoc]] LlavaOnevisionProcessor

## LlavaOnevisionImageProcessor

[[autodoc]] LlavaOnevisionImageProcessor

## LlavaOnevisionImageProcessorFast

[[autodoc]] LlavaOnevisionImageProcessorFast
    - preprocess

## LlavaOnevisionVideoProcessor

[[autodoc]] LlavaOnevisionVideoProcessor

## LlavaOnevisionForConditionalGeneration

[[autodoc]] LlavaOnevisionForConditionalGeneration
    - forward

## LlavaOnevisionModel

[[autodoc]] LlavaOnevisionModel
    - forward


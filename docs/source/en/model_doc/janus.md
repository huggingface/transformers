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

*This model was released on 2024-10-17 and added to Hugging Face Transformers on 2025-04-17 and contributed by [yaswanthgali](https://huggingface.co/yaswanthgali) and [hugosilva664](https://huggingface.co/hugosilva664).*


# Janus

[Janus](https://huggingface.co/papers/2410.13848) is an autoregressive framework that unifies multimodal understanding and generation by decoupling visual encoding into separate pathways within a unified transformer architecture. This decoupling enhances flexibility and performance, allowing each component to select the most suitable encoding methods. Experiments demonstrate that Janus surpasses previous unified models and matches or exceeds task-specific models. Janus-Pro, an advanced version, further improves performance through optimized training strategies, expanded data, and a larger model size, enhancing both multimodal understanding and text-to-image generation capabilities.

<hfoptions id="usage">
<hfoption id="JanusForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import JanusForConditionalGeneration, JanusProcessor

messages = [
    {
        "role": "user",
        "content": [
            {'type':'image', 'url': 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg'},
            {'type':"text", "text": "What is shown in this image?."}
        ]
    },
]

processor = JanusProcessor.from_pretrained("deepseek-community/Janus-Pro-1B")
model = JanusForConditionalGeneration.from_pretrained("deepseek-community/Janus-Pro-1B", dtype="auto")
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    generation_mode="text",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

output = model.generate(**inputs, max_new_tokens=40,generation_mode='text',do_sample=True)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## JanusConfig

[[autodoc]] JanusConfig

## JanusVisionConfig

[[autodoc]] JanusVisionConfig

## JanusVQVAEConfig

[[autodoc]] JanusVQVAEConfig

## JanusProcessor

[[autodoc]] JanusProcessor

## JanusImageProcessor

[[autodoc]] JanusImageProcessor

## JanusImageProcessorFast

[[autodoc]] JanusImageProcessorFast

## JanusVisionModel

[[autodoc]] JanusVisionModel
    - forward

## JanusVQVAE

[[autodoc]] JanusVQVAE
    - forward

## JanusModel

[[autodoc]] JanusModel
    - forward

## JanusForConditionalGeneration

[[autodoc]] JanusForConditionalGeneration
    - forward


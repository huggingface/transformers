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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-01-23 and contributed by [abrooks9944](https://huggingface.co/abrooks9944).*

# Granite Vision

[Granite Vision](https://www.ibm.com/new/announcements/ibm-granite-3-1-powerful-performance-long-context-and-more) introduces major upgrades over Granite 3.0, achieving top-tier benchmark performance on the Hugging Face OpenLLM Leaderboard. The entire Granite 3.1 family—dense, MoE, and guardrail models—now supports 128K token context windows, vastly expanding long-context reasoning. IBM also released new multilingual Granite Embedding models (30M–278M parameters) optimized for retrieval tasks, and the Granite Guardian series adds function-calling hallucination detection for safer tool use. All models are open source under Apache 2.0 and integrated into IBM’s watsonx.ai ecosystem and partner platforms such as Hugging Face, Ollama, and Docker.

<hfoptions id="usage">
<hfoption id="LlavaNextForConditionalGeneration">

```py
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("ibm-granite/granite-vision-3.1-2b-preview",
model = LlavaNextForConditionalGeneration.from_pretrained("ibm-granite/granite-vision-3.1-2b-preview", dtype="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- This model loads as a [`LlavaNextForConditionalGeneration`] instance.

## LlavaNextConfig

[[autodoc]] LlavaNextConfig

## LlavaNextImageProcessor

[[autodoc]] LlavaNextImageProcessor
    - preprocess

## LlavaNextProcessor

[[autodoc]] LlavaNextProcessor

## LlavaNextForConditionalGeneration

[[autodoc]] LlavaNextForConditionalGeneration
    - forward

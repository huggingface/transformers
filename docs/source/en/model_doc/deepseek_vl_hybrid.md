<!--Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-03-08 and added to Hugging Face Transformers on 2025-07-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DeepseekVLHybrid

[Deepseek-VL-Hybrid](https://huggingface.co/papers/2403.05525) [Deepseek-VL](https://huggingface.co/papers/2403.05525) is an open-source vision-language model optimized for real-world multimodal understanding. It employs a hybrid vision encoder capable of efficiently processing high-resolution images (1024×1024) while minimizing computational cost, enabling rich semantic and detail capture across diverse tasks. The model is trained on a large, diverse dataset that includes real-world content like web screenshots, PDFs, charts, and OCR data, with instruction tuning guided by a taxonomy of practical user scenarios. By integrating language model pretraining from the start to balance vision–language learning, DeepSeek-VL (available in 1.3B and 7B versions) achieves state-of-the-art performance on vision-language benchmarks while retaining strong language capabilities.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="deepseek-community/deepseek-vl-1.3b-chat", dtype="auto")
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        {"type": "text", "text": "What is shown in this image?"},
    ]},
]
pipeline(text=messages, max_new_tokens=300, return_full_text=False)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat")
model = AutoModelForImageTextToText.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat", dtype="auto")

messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        {"type": "text", "text": "What is shown in this image?"},
    ]},
]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
)
print(processor.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## DeepseekVLHybridConfig

[[autodoc]] DeepseekVLHybridConfig

## DeepseekVLHybridProcessor

[[autodoc]] DeepseekVLHybridProcessor

## DeepseekVLHybridImageProcessor

[[autodoc]] DeepseekVLHybridImageProcessor

## DeepseekVLHybridImageProcessorFast

[[autodoc]] DeepseekVLHybridImageProcessorFast

## DeepseekVLHybridModel

[[autodoc]] DeepseekVLHybridModel
    - forward

## DeepseekVLHybridForConditionalGeneration

[[autodoc]] DeepseekVLHybridForConditionalGeneration
    - forward

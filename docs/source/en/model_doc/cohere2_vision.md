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
*This model was released on 2025-07-31 and added to Hugging Face Transformers on 2025-07-31.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Command A Vision

[Command A Vision](https://cohere.com/blog/command-a-vision) s a state-of-the-art multimodal generative model optimized for enterprise use, excelling in both visual and text-based tasks. It outperforms leading models like GPT-4.1 and Llama 4 Maverick on benchmarks involving charts, diagrams, documents, and real-world imagery. The model features advanced document OCR with structured JSON outputs, strong scene understanding, and multilingual reasoning across industries such as finance, healthcare, and manufacturing. Designed for secure, efficient deployment, it runs on as little as one H100 or two A100 GPUs, enabling scalable on-premise or private enterprise applications.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="CohereLabs/command-a-vision-07-2025", dtype="auto")
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

processor = AutoProcessor.from_pretrained("CohereLabs/aya-vision-8b)
model = AutoModelForImageTextToText.from_pretrained("CohereLabs/command-a-vision-07-2025", dtype="auto")

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

## Cohere2VisionConfig

[[autodoc]] Cohere2VisionConfig

## Cohere2VisionForConditionalGeneration

[[autodoc]] Cohere2VisionForConditionalGeneration
    - forward

## Cohere2VisionModel

[[autodoc]] Cohere2VisionModel
    - forward

## Cohere2VisionImageProcessorFast

[[autodoc]] Cohere2VisionImageProcessorFast
    - preprocess

## Cohere2VisionProcessor

[[autodoc]] Cohere2VisionProcessor

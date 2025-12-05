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
*This model was released on 2025-07-26 and added to Hugging Face Transformers on 2025-12-04.*


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# Intern-S1

> *A Scientific Multimodal Foundation Model*


We introduce [**Intern-S1**](https://huggingface.co/papers/2508.15763v2), our **most advanced open-source multimodal reasoning model** to date. Intern-S1 combines **strong general-task capabilities with state-of-the-art performance on a wide range of scientific tasks**, rivaling leading closed-source commercial models.
Built upon a 235B MoE language model (Qwen3) and a 6B Vision encoder (InternViT), Intern-S1 has been further pretrained on **5 trillion tokens** of multimodal data, including over **2.5 trillion scientific-domain tokens**. This enables the model to retain strong general capabilities while excelling in specialized scientific domains such as **interpreting chemical structures, understanding protein sequences, and planning compound synthesis routes**, making Intern-S1 to be a capable research assistant for real-world scientific applications.
Features

- Strong performance across language and vision reasoning benchmarks, especially scientific tasks.

- Continuously pretrained on a massive 5T token dataset, with over 50% specialized scientific data, embedding deep domain expertise.

- Dynamic tokenizer enables native understanding of molecular formulas, protein sequences, and seismic signals.

Also, we introduce **Intern-S1-mini**, a lightweight open-source multimodal reasoning model based on the same techniques as Intern-S1. Built upon an 8B dense language model (Qwen3) and a 0.3B Vision encoder (InternViT), Intern-S1-mini has been further pretrained on 5 trillion tokens of multimodal data, including over 2.5 trillion scientific-domain tokens. This enables the model to retain strong general capabilities while excelling in specialized scientific domains such as interpreting chemical structures, understanding protein sequences, and planning compound synthesis routes, making Intern-S1-mini to be a capable research assistant for real-world scientific applications.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/intern_s1_architecture.png" alt="drawing" width="600"/>  

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/intern_s1_overview_performance.png" alt="drawing" width="600"/>

## Usage example

### Sampling Parameters

We recommend using the following hyperparameters to ensure better results

```python
top_p = 1.0
top_k = 50
min_p = 0.0
temperature = 0.8
```

### Text input

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = "internlm/Intern-S1-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", dtype="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "tell me about an interesting physical phenomenon."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=1024)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
```

### Image input

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = "internlm/Intern-S1-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", dtype="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Please describe the image explicitly."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=1024)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
```

### Video input

Please ensure that the decord video decoding library is installed via `pip install decord`. To avoid OOM, please install flash_attention and use at least 2 GPUS.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = "internlm/Intern-S1-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", dtype="auto")

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
                },
                {"type": "text", "text": "What type of shot is the man performing?"},
            ],
        }
    ]

inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    ).to(model.device, dtype=torch.float16)

generate_ids = model.generate(**inputs, max_new_tokens=1024)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
```

## InternS1Config

[[autodoc]] InternS1Config

## InternS1VisionConfig

[[autodoc]] InternS1VisionConfig

## InternS1Model

[[autodoc]] InternS1Model
    - forward

## InternS1ForConditionalGeneration

[[autodoc]] InternS1ForConditionalGeneration
    - forward

## InternS1VisionModel

[[autodoc]] InternS1VisionModel
    - forward

## InternS1Processor

[[autodoc]] InternS1Processor
    - __call__
    - decode
    - batch_decode

## InternS1Tokenizer

[[autodoc]] InternS1Tokenizer
    - __call__
    - tokenize
    - decode

## InternS1VideoProcessor

[[autodoc]] InternS1VideoProcessor

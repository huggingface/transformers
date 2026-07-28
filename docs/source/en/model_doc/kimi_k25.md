<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->


*This model was contributed to Hugging Face Transformers on 2026-07-03.*

# KimiK-2.5, KimiK-2.6, KimiK-2.7

This model class supports all three different releases: KimiK-2.5,KimiK-2.6, KimiK-2.7


## Overview

Kimi K2.5 is an open-source, native multimodal agentic model that advances practical capabilities in long-horizon coding, coding-driven design, proactive autonomous execution, and swarm-based task orchestration. The model was proposed in [Kimi K2.5: Visual Agentic Intelligence](https://www.kimi.com/en/blog/kimi-k2-5) and further improved in [Kimi K2.6: Advancing Open-Source Coding](Kimi K2.5: Visual Agentic Intelligence).

Kimi K2.5 achieves significant improvements on complex, end-to-end coding tasks, generalizing robustly across programming languages (Rust, Go, Python) and domains spanning front-end, DevOps, and performance optimization. The model is capable of transforming simple prompts and visual inputs into production-ready interfaces and lightweight full-stack workflows, generating structured layouts, interactive elements, and rich animations with deliberate aesthetic precision.

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The offical checkpoints are [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5), [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) and [moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code).


## Usage examples

<Tip warning={true}>

Note that the repositories don't yet have the correct fast tokenizer uploaded. You can get the converted processor and tokenizer from [RaushanTurganbay/kimi2.7-processor](https://huggingface.co/RaushanTurganbay/kimi2.7-processor)

</Tip> 

```python
import os
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(enable_expert_parallel=True)

processor = AutoProcessor.from_pretrained('moonshotai/Kimi-K2.6')
model = AutoModelForImageTextToText.from_pretrained(
    'moonshotai/Kimi-K2.6',
    distributed_config=distributed_config,
)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(device=model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(generated_text)

```

## Kimi_K25ImageProcessor

[[autodoc]] Kimi_K25ImageProcessor

## Kimi_K25Processor

[[autodoc]] Kimi_K25Processor

## Kimi_K25VideoProcessor

[[autodoc]] Kimi_K25VideoProcessor

## Kimi_K25Config

[[autodoc]] Kimi_K25Config

## Kimi_K25VisionConfig

[[autodoc]] Kimi_K25VisionConfig

## Kimi_K25PreTrainedModel

[[autodoc]] Kimi_K25PreTrainedModel
    - forward

## Kimi_K25VisionModel

[[autodoc]] Kimi_K25VisionModel

## Kimi_K25Model

[[autodoc]] Kimi_K25Model
    - forward

## Kimi_K25ForConditionalGeneration

[[autodoc]] Kimi_K25ForConditionalGeneration

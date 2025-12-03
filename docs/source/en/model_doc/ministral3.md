<!--Copyright 2025 the HuggingFace Team. All rights reserved.

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


# Ministral3

## Overview

A balanced model in the Ministral 3 family, Ministral 3 8B is a powerful, efficient tiny language model with vision capabilities.

This model is the instruct post-trained version, fine-tuned for instruction tasks, making it ideal for chat and instruction based use cases.

The Ministral 3 family is designed for edge deployment, capable of running on a wide range of hardware.

Key features:
- Vision: Enables the model to analyze images and provide insights based on visual content, in addition to text.
- Multilingual: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic.
- System Prompt: Maintains strong adherence and support for system prompts.
- Agentic: Offers best-in-class agentic capabilities with native function calling and JSON outputting.
- Edge-Optimized: Delivers best-in-class performance at a small scale, deployable anywhere.
- Apache 2.0 License: Open-source license allowing usage and modification for both commercial and non-commercial purposes.
- Large Context Window: Supports a 256k context window.

## Usage examples

```py
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


model_id = "mistralai/Ministral-3-3B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
)

image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device="cuda")
image_sizes = [tokenized["pixel_values"].shape[-2:]]

output = model.generate(
    **tokenized,
    image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)
```


## Ministral3Config

[[autodoc]] Ministral3Config

## Ministral3PreTrainedModel

[[autodoc]] Ministral3PreTrainedModel
    - forward

## Ministral3Model

[[autodoc]] Ministral3Model
    - forward

## Ministral3ForCausalLM

[[autodoc]] Ministral3ForCausalLM

## Ministral3ForSequenceClassification

[[autodoc]] Ministral3ForSequenceClassification

## Ministral3ForTokenClassification

[[autodoc]] Ministral3ForTokenClassification

## Ministral3ForQuestionAnswering

[[autodoc]] Ministral3ForQuestionAnswering

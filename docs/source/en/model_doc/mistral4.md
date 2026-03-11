<!--Copyright 2026 Mistral AI and the HuggingFace Team. All rights reserved.

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
*This model was released on 2026-16-02 and added to Hugging Face Transformers on 2026-16-02.*

# Mistral4 

## Overview

A balanced model in the Mistral MOE family, Mistral Small 4 is a powerful, efficient language model with vision capabilities.

This model is the instruct and reasoning post-trained version, fine-tuned for instruction tasks, making it ideal for chat and instruction based use cases as well as tasks that require deeper thinking. Mistral Small 4 was also 

The Mistral 4 family is designed for edge deployment, capable of running on a wide range of hardware.

Key features:

- Vision: Enables the model to analyze images and provide insights based on visual content, in addition to text.
- Multilingual: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic.
- System Prompt: Maintains strong adherence and support for system prompts.
- Agentic: Offers best-in-class agentic capabilities with native function calling and JSON outputting.
- Apache 2.0 License: Open-source license allowing usage and modification for both commercial and non-commercial purposes.
- Large Context Window: Supports up to a 1M context window but the recommended setting is to keep the context up to 256k.

## Usage examples

```py
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


model_id = "mistralai/Mistral-Small-4-119B-2603"

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

## Mistral4Config

[[autodoc]] Mistral4Config

## Mistral4PreTrainedModel

[[autodoc]] Mistral4PreTrainedModel
    - forward

## Mistral4Model

[[autodoc]] Mistral4Model
    - forward

## Mistral4ForCausalLM

[[autodoc]] Mistral4ForCausalLM

## Mistral4ForSequenceClassification

[[autodoc]] Mistral4ForSequenceClassification

## Mistral4ForTokenClassification

[[autodoc]] Mistral4ForTokenClassification

## Mistral4ForQuestionAnswering

[[autodoc]] Mistral4ForQuestionAnswering

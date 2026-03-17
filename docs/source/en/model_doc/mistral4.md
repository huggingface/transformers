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
*This model was released on 2026-03-16 and added to Hugging Face Transformers on 2026-03-16.*

# Mistral4 

## Overview

Mistral 4 is a powerful hybrid model with the capability of acting as both a general instruction model and a reasoning model. It unifies the capabilities of three different model families - Instruct, Reasoning ( previous called Magistral ), and Devstral - into a single, unified model.

[Mistral-Small-4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) consists of the following architectural choices:

- MoE: 128 experts and 4 active.
- 119B with 6.5B activated parameters per token.
- 256k Context Length.
- Multimodal Input: Accepts both text and image input, with text output.
- Instruct and Reasoning functionalities with Function Calls
    - Reasoning Effort configurable by request.

Mistral 4 offers the following capabilities:

- **Reasoning Mode**: Switch between a fast instant reply mode, and a reasoning thinking mode, boosting performance with test time compute when requested.
- **Vision**: Enables the model to analyze images and provide insights based on visual content, in addition to text.
- **Multilingual**: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic.
- **System Prompt**: Maintains strong adherence and support for system prompts.
- **Agentic**: Offers best-in-class agentic capabilities with native function calling and JSON outputting.
- **Speed-Optimized**: Delivers best-in-class performance and speed.
- **Apache 2.0 License**: Open-source license allowing usage and modification for both commercial and non-commercial purposes.
- **Large Context Window**: Supports a 256k context window.

## Usage examples

```py
import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration


model_id = "mistralai/Mistral-Small-4-119B-2603"

processor = AutoProcessor.from_pretrained(model_id)
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

inputs = processor.apply_chat_template(messages, return_tensors="pt", tokenize=True, return_dict=True, reasoning_effort="high")
inputs = inputs.to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512,
)[0]

# Setting `skip_special_tokens=False` to visualize reasoning trace between [THINK] [/THINK] tags.
decoded_output = processor.decode(output[len(inputs["input_ids"][0]):], skip_special_tokens=False) 
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

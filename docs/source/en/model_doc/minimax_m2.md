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


# MiniMax-M2

## Overview

MiniMax-M2 is a compact, fast, and cost-effective MoE model (230 billion total parameters with 10 billion active parameters) built for elite performance in coding and agentic tasks, all while maintaining powerful general intelligence. With just 10 billion activated parameters, MiniMax-M2 provides the sophisticated, end-to-end tool use performance expected from today's leading models, but in a streamlined form factor that makes deployment and scaling easier than ever.

For more details refer to the [release blog post](https://www.minimax.io/news/minimax-m2).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "MiniMaxAI/MiniMax-M2",
    device_map="auto",
    revision="refs/pr/52",
)

tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2", revision="refs/pr/52")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=100)

response = tokenizer.batch_decode(generated_ids)[0]

print(response)
```

## MiniMaxM2Config

[[autodoc]] MiniMaxM2Config

## MiniMaxM2Model

[[autodoc]] MiniMaxM2Model
    - forward

## MiniMaxM2ForCausalLM

[[autodoc]] MiniMaxM2ForCausalLM
    - forward

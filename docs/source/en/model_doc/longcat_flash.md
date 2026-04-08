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
*This model was released on 2025-09-01 and added to Hugging Face Transformers on 2025-09-17.*

# LongCatFlash

## Overview

The LongCatFlash model was proposed in [LongCat-Flash Technical Report](https://huggingface.co/papers/2509.01322) by the Meituan LongCat Team.
LongCat-Flash is a 560B parameter Mixture-of-Experts (MoE) model that activates 18.6B-31.3B parameters dynamically (average ~27B). The model features a shortcut-connected architecture enabling high inference speed (>100 tokens/second) and advanced reasoning capabilities.

The abstract from the paper is the following:

*We present LongCat-Flash, a 560 billion parameter Mixture-of-Experts (MoE) language model featuring a dynamic computation mechanism that activates 18.6B-31.3B parameters based on context (average ~27B). The model incorporates a shortcut-connected architecture enabling high inference speed (>100 tokens/second) and demonstrates strong performance across multiple benchmarks including 89.71% accuracy on MMLU and exceptional agentic tool use capabilities.*

Tips:

- LongCat-Flash uses a unique shortcut-connected MoE architecture that enables faster inference compared to traditional MoE models
- The model supports up to 128k context length for long-form tasks
- Dynamic parameter activation makes it computationally efficient while maintaining high performance
- Best suited for applications requiring strong reasoning, coding, and tool-calling capabilities
- The MoE architecture includes zero experts (nn.Identity modules) which act as skip connections, allowing tokens to bypass expert computation when appropriate

This model was contributed by [Molbap](https://huggingface.co/Molbap).
The original code can be found [here](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat).

## Usage examples

The model is large: you will need 2x8 H100 to run inference.

```python
# launch_longcat.py
from transformers import LongcatFlashForCausalLM, AutoTokenizer
import torch

model_id = "meituan-longcat/LongCat-Flash-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)

chat = [
      {"role": "user", "content": "Hello! What is the capital of France? What can you tell me about it?"},
]

model = LongcatFlashForCausalLM.from_pretrained(
      model_id,
      tp_plan="auto",
      dtype=torch.bfloat16,
      )

inputs = tokenizer.apply_chat_template(
      chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

outputs = model.generate(inputs, max_new_tokens=30)
print(tokenizer.batch_decode(outputs))
```

To run with TP, you will need torchrun:

```bash
torchrun  --nproc_per_node=8 --nnodes=2 --node_rank=0 | 1  --rdzv-id <an_id> --rdzv-backend c10d --rdzv-endpoint $NODE_ID:$NODE_PORT  --log-dir ./logs_longcat launch_longcat.py
```

And you'll get a nice generation:

```json
[Round 0] USER:Hello! What is the capital of France? What can you tell me about it? ASSISTANT:Hello! 😊 The capital of France is Paris, one of the most famous and beloved cities in the world. Here’s a quick overview of what makes Paris special:
1. Iconic Landmarks

    Eiffel Tower – The global symbol of France, built in 1889 for the World's Fair.
    Notre-Dame Cathedral – A masterpiece of Gothic architecture (currently under restoration after the 2019 fire).
    Louvre Museum – The world’s largest art museum, home to the Mona Lisa and Venus de Milo.
    Sacré-Cœur Basilica – A stunning white church atop Montmartre with panoramic views.
    Arc de Triomphe – Honors French military victories, with the Tomb of the Unknown Soldier beneath it.
    Champs-Élysées – A glamorous avenue leading to the Arc de Triomphe, lined with shops and cafés.

2. Culture & Arts

    Paris is the "City of Light" (La Ville Lumière), a nickname from its early adoption of street lighting and its role as a center of enlightenment.
    It’s a global hub for fashion (haute couture, Paris Fashion Week) and art (Impressionism, Picasso, Dali).
    Famous literary figures like Hemingway, Fitzgerald, and Sartre lived and wrote here.

3. Food & Cuisine

    Croissants, baguettes, macarons, and crème brûlée are just a few of its culinary delights.
    Paris has over 100 Michelin-starred restaurants and countless cozy bistros.
    The Marché d’Aligre and Rue Mouffetard are great for fresh produce and local flavors.

4. History & Politics

    Founded in the 3rd century BC by the Parisii tribe, it became a major European city under the Romans.
    The French Revolution (1789–1799) began here, leading to the fall of the monarchy.
    Today, it’s the political and economic heart of France, housing the French President’s residence (Élysée Palace) and the National Assembly.

**
```

## LongcatFlashConfig

[[autodoc]] LongcatFlashConfig

## LongcatFlashPreTrainedModel

[[autodoc]] LongcatFlashPreTrainedModel
    - forward

## LongcatFlashModel

[[autodoc]] LongcatFlashModel
    - forward

## LongcatFlashForCausalLM

[[autodoc]] LongcatFlashForCausalLM

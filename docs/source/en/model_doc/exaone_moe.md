<!--Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-31 and added to Hugging Face Transformers on 2026-01-25.*

# EXAONE MoE

## Overview

**[K-EXAONE](https://github.com/LG-AI-EXAONE/K-EXAONE)** model is a large-scale multilingual language model developed by LG AI Research. Built using a Mixture-of-Experts architecture named `EXAONE-MoE`, K-EXAONE features **236 billion total** parameters, with **23 billion active** during inference. Performance evaluations across various benchmarks demonstrate that K-EXAONE excels in reasoning, agentic capabilities, general knowledge, multilingual understanding, and long-context processing.

### Key Features

- **Architecture & Efficiency:** Features a 236B fine-grained MoE design (23B active) optimized with **Multi-Token Prediction (MTP)**, enabling self-speculative decoding that boosts inference throughput by approximately 1.5x.
- **Long-Context Capabilities:** Natively supports a **256K context window**, utilizing a **3:1 hybrid attention** scheme with a **128-token sliding window** to significantly minimize memory usage during long-document processing.
- **Multilingual Support:** Covers 6 languages: Korean, English, Spanish, German, Japanese, and Vietnamese. Features a redesigned **150k vocabulary** with **SuperBPE**, improving token efficiency by ~30%.
- **Agentic Capabilities:** Demonstrates superior tool-use and search capabilities via **multi-agent strategies.**
- **Safety & Ethics:** Aligned with **universal human values**, the model uniquely incorporates **Korean cultural and historical contexts** to address regional sensitivities often overlooked by other models. It demonstrates high reliability across diverse risk categories.

For more details, please refer to the [technical report](https://www.lgresearch.ai/data/cdn/upload/K-EXAONE_Technical_Report.pdf) and [GitHub](https://huggingface.co/collections/LGAI-EXAONE/k-exaone).

All model weights including quantized version are available at [Huggingface Collections](https://huggingface.co/collections/LGAI-EXAONE/k-exaone).

## Model Details

### Model Configuration of K-EXAONE

- Number of Parameters: 236B in total and 23B activated
- Number of Parameters (without embeddings): 234B
- Hidden Dimension: 6,144
- Number of Layers: 48 Main layers + 1 MTP layers
  - Hybrid Attention Pattern: 12 x (3 Sliding window attention + 1 Global attention)
- Sliding Window Attention
  - Number of Attention Heads: 64 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - Sliding Window Size: 128
- Global Attention
  - Number of Attention Heads: 64 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - No Rotary Positional Embedding Used (NoPE)
- Mixture of Experts:
  - Number of Experts: 128
  - Number of Activated Experts: 8
  - Number of Shared Experts: 1
  - MoE Intermediate Size: 2,048
- Vocab Size: 153,600
- Context Length: 262,144 tokens
- Knowledge Cutoff: Dec 2024 (2024/12)

## Usage Tips

### Reasoning mode

For tasks that require accurate results, you can run the K-EXAONE model in reasoning mode as below.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/K-EXAONE-236B-A23B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Which one is bigger, 3.9 vs 3.12?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,   # skippable (default: True)
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=16384,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### Non-reasoning mode

For tasks where latency matters more than accuracy, you can run the K-EXAONE model in non-reasoning mode as below.

```python
messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Explain how wonderful you are"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### Agentic tool use

For your AI-powered agent, you can leverage K-EXAONE’s tool calling capability. 
The K-EXAONE model is compatible with both OpenAI and HuggingFace tool calling specifications. 
The example below demonstrates tool calling using HuggingFace’s docstring-to-tool-schema utility.

Please check the [example file](https://github.com/LG-AI-EXAONE/K-EXAONE/blob/main/examples/example_output_search.txt) for an example of a search agent conversation using K-EXAONE.

```python
from transformers.utils import get_json_schema

def roll_dice(max_num: int):
    """
    Roll a dice with the number 1 to N. User can select the number N.

    Args:
        max_num: The maximum number on the dice.
    """
    return random.randint(1, max_num)

tool_schema = get_json_schema(roll_dice)
tools = [tool_schema]

messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Roll a D20 twice and sum the results."}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    tools=tools,
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=16384,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

## ExaoneMoeConfig

[[autodoc]] ExaoneMoeConfig

## ExaoneMoeModel

[[autodoc]] ExaoneMoeModel
    - forward

## ExaoneMoeForCausalLM

[[autodoc]] ExaoneMoeForCausalLM
    - forward

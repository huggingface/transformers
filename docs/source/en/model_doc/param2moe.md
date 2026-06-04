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
*This model was contributed to Hugging Face Transformers on 2026-02-10.*

# Param2MoE

## Overview

Param2MoE was released by the [BharatGen AI](https://huggingface.co/bharatgenai) team as **Param-2-17B-MoE-A2.4B**, a Hybrid Mixture-of-Experts language model with 17B total parameters and only 2.4B active per token. It is pretrained from scratch on ~22 trillion tokens across two phases, with an emphasis on linguistic diversity — supporting English, Hindi, and 21 Indian languages. The model ships as an early post-training checkpoint with reasoning, tool calling, math, and code capabilities.

The original model can be found [here](https://huggingface.co/bharatgenai/Param2-17B-A2.4B-Thinking).

Tips:

- **Expert routing**: 2 shared experts are always active alongside 6 dynamically routed experts selected from a pool of 64 per token.
- **Memory**: Loading the full model requires ~34 GB VRAM in bfloat16.
- **Context**: Maximum supported length is 4096 tokens.
- **Custom code**: Pass `trust_remote_code=True` when loading with `AutoModelForCausalLM`.
- **Decoding**: Set `skip_special_tokens=False` to preserve `<think>...</think>` reasoning tags in Thinking checkpoint outputs. Use `do_sample=False` for deterministic/evaluation runs.
- **Safety**: The model has not undergone RLHF or safety alignment — fine-tune and evaluate before production use.

## Usage examples

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from parsers import parse_model_output

model_name = "bharatgenai/Param2-17B-A2.4B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

conversation = [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "What is the BharatGen Mission?"}
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

generated_tokens = output[0][inputs.shape[-1]:]

# IMPORTANT: skip_special_tokens=False
generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=False
)

parsed = parse_model_output(generated_text)

print("\n========== RAW ==========\n", generated_text)
print("\n========== REASONING ==========\n", parsed["reasoning"])
print("\n========== TOOL CALLS ==========\n", parsed["tool_calls"])
print("\n========== FINAL ANSWER ==========\n", parsed["final_answer"])
```

## Param2MoEConfig

[[autodoc]] Param2MoEConfig

## Param2MoEPreTrainedModel

[[autodoc]] Param2MoEPreTrainedModel
    - forward

## Param2MoEModel

[[autodoc]] Param2MoEModel
    - forward

## Param2MoEForCausalLM

[[autodoc]] Param2MoEForCausalLM
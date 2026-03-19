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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-10-07.*

# Lfm2Moe

## Overview

LFM2-MoE is a Mixture-of-Experts (MoE) variant of [LFM2](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38). The LFM2 family is optimized for on-device inference by combining short‑range, input‑aware gated convolutions with grouped‑query attention (GQA) in a layout tuned to maximize quality under strict speed and memory constraints.

LFM2‑MoE keeps this fast backbone and introduces sparse MoE feed‑forward networks to add representational capacity without significantly increasing the active compute path. The first LFM2-MoE release is LFM2-8B-A1B, with 8.3B total parameters and 1.5B active parameters. The model excels in quality (comparable to 3-4B dense models) and speed (faster than other 1.5B class models).

## Example

The following example shows how to generate an answer using the `AutoModelForCausalLM` class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM2-8B-A1B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

## Lfm2MoeConfig

[[autodoc]] Lfm2MoeConfig

## Lfm2MoeForCausalLM

[[autodoc]] Lfm2MoeForCausalLM

## Lfm2MoeModel

[[autodoc]] Lfm2MoeModel
    - forward

## Lfm2MoePreTrainedModel

[[autodoc]] Lfm2MoePreTrainedModel
    - forward

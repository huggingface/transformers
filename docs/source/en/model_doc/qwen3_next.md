<!--Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-10.*

## Overview

The Qwen3-Next series represents our next-generation foundation models, optimized for extreme context length and large-scale parameter efficiency.
The series introduces a suite of architectural innovations designed to maximize performance while minimizing computational cost:

- **Hybrid Attention**: Replaces standard attention with the combination of **Gated DeltaNet** and **Gated Attention**, enabling efficient context modeling.
- **High-Sparsity MoE**: Achieves an extreme low activation ratio as 1:50 in MoE layers — drastically reducing FLOPs per token while preserving model capacity.
- **Multi-Token Prediction(MTP)**: Boosts pretraining model performance, and accelerates inference.
- **Other Optimizations**: Includes techniques such as **zero-centered and weight-decayed layernorm**, **Gated Attention**, and other stabilizing enhancements for robust training.

Built on this architecture, we trained and open-sourced Qwen3-Next-80B-A3B — 80B total parameters, only 3B active — achieving extreme sparsity and efficiency.

Despite its ultra-efficiency, it outperforms Qwen3-32B on downstream tasks — while requiring **less than 1/10 of the training cost**.
Moreover, it delivers over **10x higher inference throughput** than Qwen3-32B when handling contexts longer than 32K tokens.

For more details, please visit our blog [Qwen3-Next](qwen3_next) ([blog post](https://qwenlm.github.io/blog/qwen3_next/)).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

## Qwen3NextConfig

[[autodoc]] Qwen3NextConfig

## Qwen3NextModel

[[autodoc]] Qwen3NextModel
    - forward

## Qwen3NextForCausalLM

[[autodoc]] Qwen3NextForCausalLM
    - forward

## Qwen3NextForSequenceClassification

[[autodoc]] Qwen3NextForSequenceClassification
    - forward

## Qwen3NextForQuestionAnswering

[[autodoc]] Qwen3NextForQuestionAnswering
    - forward

## Qwen3NextForTokenClassification

[[autodoc]] Qwen3NextForTokenClassification
    - forward

<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not
be rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-03-05.*

# DeepSeek-V3.2

## Overview

DeepSeek-V3.2 extends the DeepSeek-V3 MLA + MoE stack with a Dynamic Sparse Attention (DSA) indexer. This integration
adds native `transformers` support for the `deepseek_v32` architecture used by the official
`deepseek-ai/DeepSeek-V3.2` checkpoints.

The original code can be found in the model repository
[here](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/inference).

## Usage example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = "/tmp/deepseek-v32"
offload_dir = "/tmp/deepseek-v32-offload"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3.2",
    cache_dir=cache_dir,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder=offload_dir,
    dtype=torch.bfloat16,
).eval()

# The official tokenizer does not currently ship a chat template, so pass a prompt string directly.
inputs = tokenizer("See", return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False))
```

## DeepseekV32Config

[[autodoc]] DeepseekV32Config

## DeepseekV32PreTrainedModel

[[autodoc]] DeepseekV32PreTrainedModel
    - forward

## DeepseekV32Model

[[autodoc]] DeepseekV32Model
    - forward

## DeepseekV32ForCausalLM

[[autodoc]] DeepseekV32ForCausalLM
    - forward

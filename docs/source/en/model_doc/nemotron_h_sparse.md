<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-15 and added to Hugging Face Transformers on 2026-04-22.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# NemotronHSparse

[NemotronHSparse](https://huggingface.co/papers/2504.03624) is the sparse (Mixture-of-Experts) variant of
the [NemotronH](./nemotron_h) hybrid architecture, covering the **Nemotron-3** lineage. Each decoder layer is
either a Mamba2 block (`M`) or an attention block (`*`), with a routed MoE feed-forward tail. Dense (MLP)
checkpoints live under [`NemotronHDense`](./nemotron_h_dense).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## NemotronHSparseConfig

[[autodoc]] NemotronHSparseConfig

## NemotronHSparseModel

[[autodoc]] NemotronHSparseModel
    - forward

## NemotronHSparseForCausalLM

[[autodoc]] NemotronHSparseForCausalLM
    - forward

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
*This model was contributed to Hugging Face Transformers on 2026-09-02.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
        <img alt="Expert parallelism" src="https://img.shields.io/badge/Expert%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Hy4-Preview

## Overview

Hy4-Preview is a 780B-parameter mixture-of-experts language model that activates 49B parameters per
token. Each MoE layer holds 256 routed experts plus one always-active shared expert and routes every
token to 8 of them. The context window is 1M tokens.

The architecture combines four features:

- **Multi-head Latent Attention (MLA)** compresses keys and values into a low-rank latent
  (`kv_lora_rank`) that `kv_b_proj` expands back to one key/value per query head.
- **DeepSeek Sparse Attention (DSA)** selects `index_topk` keys per query with a lightweight indexer.
  Following [IndexShare](https://huggingface.co/papers/2603.12201), only the layers marked `"full"`
  in `indexer_types` run an indexer; `"shared"` layers reuse the previous full layer's selection.
- **Gated MLA with learnable attention sinks**, where each head owns a sink logit that participates
  in the softmax and contributes no value, as in [GPT-OSS](./gpt_oss).
- **Independent Hyper-Connections (iHC)** replace the plain residual path with `hc_mult` parallel
  residual streams that are collapsed before, and redistributed after, every sublayer.

The implementation does not execute the multi-token prediction (MTP) layers. Released checkpoints
keep those weights so that other runtimes can use them for speculative decoding; they are ignored
at load time.

## Usage example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tencent/Hy4-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The full checkpoint does not fit on a single accelerator. Shard it with tensor parallelism, or place
each expert group on its own device with expert parallelism:

```python
from transformers import AutoModelForCausalLM, DistributedConfig

# Tensor parallel: launch with `torchrun --nproc-per-node <world_size>`.
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, distributed_config=DistributedConfig(tp_size=16)
)

# Expert parallel: routed experts are split along the expert axis.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=16, enable_expert_parallel=True),
)
```

Expert parallelism is inference-only, because the routed-expert all-reduce has no backward pass.

## HYV4Config

[[autodoc]] HYV4Config

## HYV4Model

[[autodoc]] HYV4Model
    - forward

## HYV4ForCausalLM

[[autodoc]] HYV4ForCausalLM
    - forward

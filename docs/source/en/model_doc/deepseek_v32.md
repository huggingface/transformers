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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FP8" src="https://img.shields.io/badge/FP8-4d8a4d?style=flat">
    </div>
</div>

# DeepSeek-V3.2

## Overview

[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) is an experimental release from DeepSeek-AI that introduces **DeepSeek Sparse Attention (DSA)**, a trainable, fine-grained sparse attention mechanism designed to improve training and inference efficiency in long-context scenarios. It is built directly on top of [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus): the model keeps the same 685B-parameter Mixture-of-Experts (MoE) backbone and Multi-head Latent Attention (MLA), and is obtained through continued training that adds the sparse-attention indexer while deliberately aligning the training distribution with V3.1-Terminus so the two models can be compared head-to-head.

The work was later extended in the [DeepSeek-V3.2 technical report](https://huggingface.co/papers/2512.02556), *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, which pairs DSA with a scalable reinforcement-learning framework and reports gold-medal level results on competition math (IMO) and competitive programming (IOI) benchmarks.

The abstract from the DeepSeek-V3.2-Exp release is the following:

*We introduce DeepSeek-V3.2-Exp, an experimental version of our model that incorporates DeepSeek Sparse Attention (DSA) to explore and validate optimizations for training and inference efficiency in long-context scenarios. DeepSeek Sparse Attention achieves fine-grained sparse attention for the first time with minimal impact on model output quality. Built upon DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp delivers substantially improved efficiency in both training and inference, especially in long-context settings, while maintaining virtually identical benchmark performance.*

### DeepSeek Sparse Attention (DSA)

DSA reduces the quadratic cost of attention over long sequences by attending only to a selected subset of past tokens. It has two components:

1. **Lightning indexer.** A lightweight, low-head-count scoring module computes an *index score* between each query and every preceding key. In the reference implementation it runs in FP8 with a Hadamard (`rotate_activation`) transform; because the transform is orthogonal (`Hq·Hk = q·k`) and FP8 is only a precision optimization, the transformers port computes the same scores directly in bf16/fp32, keeping the indexer cheap relative to the main attention.
2. **Fine-grained token selection.** For each query the indexer keeps the top-`index_topk` (2048 by default) tokens, and main MLA attention is then computed only over those tokens via an additive mask. This turns the per-query attention cost from `O(L)` to `O(index_topk)` for long sequences when using `flash_mla`, which is not supported yet 😉.

The indexer keeps its own small per-token key cache (single-head, `index_head_dim`) alongside the main K/V cache. In transformers this lives on a dedicated cache layer — [`DynamicIndexedLayer`] for growing caches and [`StaticIndexedLayer`] for static / `torch.compile` caches — and is updated through `past_key_values.update_indexer()`.

In DeepSeek-V3.2 **every layer runs its own indexer** — there is no cross-layer top-k sharing.

> [!NOTE]
> **The MLA query LoRA path (`q_lora_rank`) is required.** The indexer scores queries from the low-rank query latent `q_a_layernorm(q_a_proj(x))` (its `wq_b` projection is sized by `q_lora_rank`), so the model always uses the LoRA query path and `q_lora_rank` must be set — the released checkpoint uses `1536`. The optional non-LoRA `q_proj` path that [DeepSeek-V3](./deepseek_v3) exposes for `q_lora_rank=None` is **not supported** here: without the query latent there is nothing for the indexer to consume.

## Usage examples

DeepSeek-V3.2-Exp is distributed as an FP8 checkpoint. The indexer projections are kept out of FP8 quantization, since the checkpoint stores them in bf16/fp32:

```python
from transformers import FineGrainedFP8Config, AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-V3.2-Exp"
quantization_config = FineGrainedFP8Config(
    modules_to_not_convert=["model.layers.*.mlp.gate.*", "*.self_attn.indexer.weights_proj.*"],
    weight_block_size=(128, 128),
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("What are we having for dinner?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The original code can be found [here](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).

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

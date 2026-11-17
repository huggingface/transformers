<!--Copyright 2026 Translated SRL and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-11-12 and contributed to Hugging Face Transformers on 2026-11-16.*


# Lara Mini

## Overview

Lara Mini is a multilingual base language model by [Translated](https://translated.com) with 7.2B total and 0.6B
active parameters per token. It is a hybrid of linear-time recurrent layers, sliding-window attention and a few global
attention layers, each followed by a fine-grained mixture of experts.

For more details, see the [Lara Mini Base model card](https://huggingface.co/translated/Lara-4-Mini-Base) and
the [technical report](https://huggingface.co/papers/2611.16956). The weights are released under the CC BY-NC-SA 4.0
license (non-commercial use) in a gated repository: request access on the model page, then log in with
`hf auth login`.

This model was contributed by [cvalgian](https://huggingface.co/cvalgian).

## Architecture

Every layer has one token mixer, chosen per layer by `layer_types`, followed by a mixture-of-experts block. The released
model repeats the pattern gated delta net, sliding-window attention, gated delta net, global attention, sliding-window
attention over its 34 layers.

- `"linear_attention"` layers are gated delta nets (GDN), as in Qwen3.5. The published checkpoints fuse their input
  projections into one `in_proj` and call their gated output RMSNorm `out_norm`; these weights are converted when the
  model is loaded and converted back when it is saved.
- `"sliding_attention"` layers attend to the last `sliding_window` tokens (the current one included) with rotary
  position embeddings.
- `"full_attention"` layers attend to the whole sequence without position embeddings. The sliding-window and GDN
  layers carry position information.
- Both attention kinds use grouped-query attention with an RMSNorm on each query and key head.
- The mixture of experts routes each token to 4 of 128 experts with a float32 softmax router and adds an ungated shared
  expert.

The RMSNorms apply their weight in float32. Install [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) for fast GDN kernels; without them a PyTorch
implementation is used.

## Example

The following example shows how to generate text using the `AutoModelForCausalLM` class.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "translated/Lara-4-Mini-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

inputs = tokenizer("La capitale d'Italia è", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Fine-tuning with LoRA

Since the GDN input projection is split when loading, LoRA adapters target `in_proj_qkv`, `in_proj_z`, `in_proj_b` and
`in_proj_a`, alongside `out_proj`, the attention `q_proj`/`k_proj`/`v_proj`/`o_proj` and the shared expert
`gate_proj`/`up_proj`/`down_proj`. The routed experts are 3D parameters: target them with `target_parameters`. Pass
`output_router_logits=True` to add the router load-balancing loss used in pretraining.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
)
model = get_peft_model(model, config)
```

Adapters trained with the Hub remote code (`trust_remote_code=True`) target the fused `linear_attn.in_proj` instead.
They load here only partially and without a warning: the `in_proj` LoRA weights are dropped. Convert them by copying
`lora_A` to the four projections and splitting `lora_B` by rows into
`[2 * key_dim + value_dim, value_dim, num_v_heads, num_v_heads]`. The word embeddings are tied, so don't merge an
adapter on `embed_tokens` or `lm_head` without untying them first.

## LaraMiniConfig

[[autodoc]] LaraMiniConfig

## LaraMiniModel

[[autodoc]] LaraMiniModel
    - forward

## LaraMiniForCausalLM

[[autodoc]] LaraMiniForCausalLM
    - forward

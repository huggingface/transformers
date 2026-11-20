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

> [!TIP]
> Set `use_kernels=True` in [`~PreTrainedModel.from_pretrained`] to replace supported layers with optimized kernels from the Hub. Refer to [Loading kernels](../kernel_doc/loading_kernels) to learn more.

## Usage tips

- The published checkpoints fuse the gated delta net (GDN) input projections into one `in_proj` and call their gated
  output norm `out_norm`. They are converted when the model is loaded and converted back when it is saved.
- The GDN layers use fast GPU kernels when available: Hub kernels when loading with `use_kernels=True`, otherwise the
  [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and
  [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) packages if installed. Without them, they fall back to a
  slower PyTorch implementation.

## Example

The following example shows how to generate text using the `AutoModelForCausalLM` class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "translated/Lara-4-Mini-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

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

> [!WARNING]
> Adapters trained with the Hub remote code (`trust_remote_code=True`) target the fused `linear_attn.in_proj` instead.
> They load here only partially and without a warning: the `in_proj` LoRA weights are dropped. Convert them by copying
> `lora_A` to the four projections and splitting `lora_B` by rows into
> `[2 * key_dim + value_dim, value_dim, num_v_heads, num_v_heads]`. The word embeddings are tied, so don't merge an
> adapter on `embed_tokens` or `lm_head` without untying them first.

## LaraMiniConfig

[[autodoc]] LaraMiniConfig

## LaraMiniModel

[[autodoc]] LaraMiniModel
    - forward

## LaraMiniForCausalLM

[[autodoc]] LaraMiniForCausalLM
    - forward

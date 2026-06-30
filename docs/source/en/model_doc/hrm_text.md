<!--Copyright 2026 The Sapient AI Authors and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-06-26 and contributed to Hugging Face Transformers on 2026-05-18.*

# HRM-Text

## Overview

HRM-Text is the improved autoregressive language-modeling variant of the Hierarchical Reasoning Model
(HRM, [Hierarchical Reasoning Model](https://huggingface.co/papers/2506.21734)) by the Sapient AI team.
It is a base model that uses a *hierarchical recurrent* forward — two transformer stacks (`H` for slow,
abstract planning, and `L` for fast, detailed computation) are reused inside a nested recurrence:

```
for h in range(H_cycles):
    for l in range(L_cycles):
        z_L = L(z_L + z_H)
    z_H = H(z_H + z_L)
```

Architectural traits:

- **PrefixLM attention**: instruction tokens attend bidirectionally, response tokens attend
  causally. Controlled by `config.prefix_lm` (default `True`); see [4D-masks blog](https://huggingface.co/blog/poedator/4d-masks) /
  [FlexAttention blog](https://pytorch.org/blog/flexattention/) for the canonical form.
- **Per-head sigmoid output gate** applied to the attention output before `o_proj`
  (Qwen3-Next-style; see [`Qwen3NextAttention`](./qwen3_next)). Legacy checkpoints stored as
  a single fused `gqkv_proj` are split into `gate_proj` / `q_proj` / `k_proj` / `v_proj` at
  load time by the registered HRM-Text checkpoint conversion mapping.
- **Parameterless RMSNorm** — `F.rms_norm` with no learnable scale.
- **`L_bp_cycles`** — the *k-step grad trick* from HRM. At training time, only the trailing
  `L_bp_cycles[i]` of the `L_cycles` low-level iterations propagate gradients;
  earlier iterations run under `torch.no_grad()` so their activations are not
  stored. No effect at inference.

## Usage

HRM-Text-1B is a **base language model**. It does not ship a `chat_template` and
`apply_chat_template` is intentionally not supported for this release — the prompt
format used during pre-training is still evolving, and an instruction-tuned variant with
a stable chat template will follow in a separate release. Drive the base model through
plain `AutoTokenizer` + `AutoModelForCausalLM.generate(...)`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")
model = AutoModelForCausalLM.from_pretrained(
    "sapientinc/HRM-Text-1B", device_map="auto",
)

inputs = tokenizer("The quick brown fox", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Attention backends

`"sdpa"` is the default, and is the right choice for most workloads. `"flex_attention"`
is supported and pays off at long context — but it carries a fixed BlockMask construction
cost per forward that does not amortise to the win you might expect from HRM-Text's
recurrent stack reuse. Indicative prefill latency on a single H100 with the released
1.2B base checkpoint and the default `H_cycles=2`, `L_cycles=3`:

| seq_len | sdpa     | flex_attention | recommendation |
|---------|----------|----------------|----------------|
|   64    |  41 ms   |  70 ms         | sdpa           |
|  256    |  41 ms   |  70 ms         | sdpa           |
| 1024    |  42 ms   |  69 ms         | sdpa           |
| 2048    |  85 ms   |  78 ms         | flex (≈ 1.1x)  |

So pick the backend by the workload:

```python
# Default — short / medium context
model = AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B", device_map="auto")

# Long context (≥ 2K tokens) — FlexAttention's per-block sparsity overtakes SDPA
model = AutoModelForCausalLM.from_pretrained(
    "sapientinc/HRM-Text-1B", device_map="auto", attn_implementation="flex_attention",
)
```

Both backends produce equivalent logits (verified top-1 100% match end-to-end against
the torch reference). `"eager"` is supported and produces the same logits, but is rarely
the fastest option on modern hardware. Its main use is `output_attentions=True` —
SDPA / FlexAttention do not return per-head attention weights, so passes that need them
for analysis or visualisation should run with `attn_implementation="eager"`.

> [!WARNING]
> Any FlashAttention variation — FA 2/3/4 and HF Hub kernel implementations that may
> not follow the `flash_attention_*` naming convention — is rejected by [`HrmTextModel`]
> at init whenever `config.prefix_lm=True` (the default). FA backends only accept causal
> vs. non-causal masks and cannot represent the PrefixLM 4-D overlay. Use `"sdpa"`
> (default) or `"flex_attention"` for PrefixLM. Setting `config.prefix_lm=False` makes
> the mask pure causal and re-enables FA — useful for causal-only fine-tuning or
> inference paths where FA is the fastest option.

### PrefixLM training

For supervised fine-tuning that respects the instruction / response boundary, emit
`token_type_ids` from the data collator alongside `input_ids` — positions inside the
instruction get `1`, response and padding get `0`. The model treats every position with
`token_type_ids == 1` as part of a single bidirectional block; everything else stays
causal:

```python
import torch

def collate_prefixlm(batch, pad_token_id=0, ignore_label_id=-100):
    """`batch[i] = {"instruction_ids": [...], "response_ids": [...]}`."""
    full_ids = [b["instruction_ids"] + b["response_ids"] for b in batch]
    prefix_lens = [len(b["instruction_ids"]) for b in batch]
    max_len = max(len(ids) for ids in full_ids)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, ignore_label_id)
    attention_mask = torch.zeros_like(input_ids)

    for i, (ids, plen) in enumerate(zip(full_ids, prefix_lens)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        token_type_ids[i, :plen] = 1                      # bidirectional prefix
        labels[i, plen : len(ids)] = input_ids[i, plen : len(ids)]  # loss on response only
        attention_mask[i, : len(ids)] = 1
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

See [`HrmTextModel.forward`] for the accepted shape.

## HrmTextConfig

[[autodoc]] HrmTextConfig

## HrmTextModel

[[autodoc]] HrmTextModel
    - forward

## HrmTextForCausalLM

[[autodoc]] HrmTextForCausalLM
    - forward

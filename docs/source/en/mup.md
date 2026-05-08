<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# μP (Maximal Update Parametrization)

[μP](https://hf.co/papers/2203.03466) is a width-aware reparameterization of weight initialization, the forward pass, and the optimizer learning rate that keeps activation, logit, and update magnitudes bounded as the hidden size grows. In practice, hyperparameters tuned on a small "base" model transfer to a wider model without retuning.

Enable μP by setting `mup=True` and `mup_base_width` on the model config. [`PreTrainedModel`] then rewrites the output projection ([`~integrations.MuReadout`]) and the attention scale, and rescales hidden weight initialization, automatically. The flags are saved to the config, so the architecture is rebuilt identically on [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModelForCausalLM, LlamaConfig

config = LlamaConfig(
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=8,
    mup=True,
    mup_base_width=128,
)
model = AutoModelForCausalLM.from_config(config)
```

Use [`~integrations.build_mup_param_groups`] to build optimizer parameter groups that apply the μP learning-rate rule.

```py
import torch
from transformers.integrations import build_mup_param_groups

optimizer = torch.optim.AdamW(build_mup_param_groups(model, lr=1e-3))
```

## Scaling rules

With base hidden size `d0` and current hidden size `d`, define `m = d / d0`. The Adam recipe (matching [`mup.MuAdam`](https://github.com/microsoft/mup) on the matrix-like / vector-like split) is:

| Parameter category                                | Forward            | Init std       | Adam learning rate  |
|---------------------------------------------------|--------------------|----------------|---------------------|
| matrix-like (hidden weights, both dims width)     | x1                 | `std / sqrt(m)`| `lr / m`            |
| vector-like (readout, embeddings, biases, LN)     | x1                 | unchanged      | `lr`                |
| readout via [`~integrations.MuReadout`]           | input `x / m`      | unchanged      | `lr`                |

Attention logits use `1/d_head` instead of `1/sqrt(d_head)`.

A "matrix-like" weight is a 2-D weight whose fan-in and fan-out both scale with width — every linear projection inside a transformer block. The readout weight is "vector-like" because its output dimension is the (finite) vocabulary size; bounded logits at any width come from the input rescale inside [`~integrations.MuReadout`], not from a learning-rate adjustment.

## μTransfer

1. Tune the learning rate (and other hyperparameters) on the base model (`hidden_size = mup_base_width`).
2. Increase `hidden_size` while keeping `mup_base_width` fixed and reuse the same hyperparameters.

## Reproducing the paper

The two empirical contracts of μP can be checked end-to-end with [`examples/pytorch/mup_demo.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/mup_demo.py).

**Coordinate check.** Per-module mean of `|activation|` should be approximately width-invariant under μP and fan out under standard parametrization (SP, the default training regime that μP replaces).

```bash
python examples/pytorch/mup_demo.py --mode coord --widths 64 128 256 512
```

**Learning-rate transfer.** The loss-vs-learning-rate curves at width `mup_base_width` should overlap the curves at wider widths under μP, and shift under SP. This is what makes the optimal learning rate transfer.

```bash
python examples/pytorch/mup_demo.py --mode lr-transfer \
    --widths 64 128 256 --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 1e-1
```

## Coordinate check from Python

[`~integrations.coord_check`] is the underlying utility used by the demo script.

```py
from transformers.integrations import coord_check

records = coord_check(
    model_factory=lambda w: build_model(width=w),
    widths=[128, 256, 512],
    batch=batch,
    n_steps=4,
    lr=1e-3,
)
```

## Compatibility

μP requires the model's attention modules to expose `module.scaling` and `module.head_dim`, and the model class to implement [`~PreTrainedModel.get_output_embeddings`] and [`~PreTrainedModel.set_output_embeddings`]. These are the conventions used by all attention backends in Transformers and by every causal-LM head, so most modern models are supported out of the box. Models that bypass these conventions (for example, attention modules with an inlined scale literal) need a per-model override.

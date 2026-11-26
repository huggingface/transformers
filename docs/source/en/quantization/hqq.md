<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# HQQ

[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq/) optimizes its quantization parameters from the weights alone, so it doesn't need calibration data. Use it to quantize a model to as few as 1 or 2 bits while it loads.

The table below summarizes what HQQ supports in Transformers.

| | |
|---|---|
| Config | [`HqqConfig`] |
| `quant_method` | `hqq` |
| Workflow | Quantize while loading |
| Bits | 1, 2, 3, 4, 8 |
| Hardware | CUDA, XPU |
| Save | ✅ |
| Fine-tune | ✅ |
| Auto-compile | ❌ |

## Install HQQ

Install hqq 0.2.1 or later.

```bash
pip install -U hqq
```

## Quantize a model with HQQ

Create a [`HqqConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. The example below quantizes every linear layer to 4 bits in groups of 64 weights.

```py
import torch
from transformers import AutoModelForCausalLM, HqqConfig

quantization_config = HqqConfig(nbits=4, group_size=64)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

## Configure HQQ

These [`HqqConfig`] parameters change how the weights are quantized.

| Parameter | Default | Effect |
|---|---|---|
| `nbits` | `4` | Number of bits per weight. Supports 8, 4, 3, 2, and 1. |
| `group_size` | `64` | Number of weights that share a scale and zero point. Smaller groups are more accurate and use more memory. |
| `axis` | `1` | Axis to group weights along, either 0 or 1. The faster torchao and BitBlas backends work only with `1`. |
| `view_as_float` | `False` | Stores the quantized weights as a float dtype, for distributed training. |
| `skip_modules` | `["lm_head"]` | Layer types to leave unquantized. |

## Quantize layer types with different settings

Pass `dynamic_config` to quantize each type of linear layer with its own settings. Each key is a *layer tag*, the module name with `model`, `layers`, and the layer number removed. For example, `model.layers.3.self_attn.q_proj` has the tag `self_attn.q_proj`.

The example below quantizes the attention projections to 4 bits and the MLP projections to 3 bits.

```py
q4_config = {"nbits": 4, "group_size": 64}
q3_config = {"nbits": 3, "group_size": 32}
quantization_config = HqqConfig(dynamic_config={
    "self_attn.q_proj": q4_config,
    "self_attn.k_proj": q4_config,
    "self_attn.v_proj": q4_config,
    "self_attn.o_proj": q4_config,
    "mlp.gate_proj": q3_config,
    "mlp.up_proj": q3_config,
    "mlp.down_proj": q3_config,
})
```

A tag applies to that layer type in every decoder layer, so you can't give one decoder layer different settings from the others. Linear layers whose tags aren't in `dynamic_config` stay unquantized. The same tags are used to match `skip_modules`.

## Choose a backend

HQQ runs the quantized layers through a backend from the hqq library. The PyTorch and CUDA backends work with PEFT fine-tuning and older GPUs. Switch backends with `HQQLinear.set_backend`.

```py
from hqq.core.quantize import HQQBackend, HQQLinear

HQQLinear.set_backend(HQQBackend.PYTORCH)
```

For faster 4-bit inference, convert the quantized model to a fused kernel backend, like torchao, after loading it. The torchao backend needs `axis=1`, which is the default.

```py
from hqq.utils.patching import prepare_for_inference

prepare_for_inference(model, backend="torchao_int4")
```

See the hqq [Backend](https://github.com/mobiusml/hqq/#backend) guide for the available backends.

## Limitations

- HQQ can't offload to the CPU or disk. Loading with a `device_map` that includes `"cpu"` or `"disk"` raises a `ValueError`.
- Layer tags can't target one decoder layer, so `dynamic_config` and `skip_modules` apply to a layer type across the whole model.

## Resources

- [Half-Quadratic Quantization of Large Machine Learning Models](https://dropbox.github.io/hqq_blog/) blog post
- [hqq](https://github.com/mobiusml/hqq/) repository

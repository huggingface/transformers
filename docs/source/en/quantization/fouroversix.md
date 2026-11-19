<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Four Over Six

[Four Over Six](https://github.com/mit-han-lab/fouroversix) is a variant of [NVFP4](./nvfp4) quantization that scales the largest value in each block of 16 FP4 values to either 4 or 6, picking whichever gives the lower quantization error. A scale of 6 uses the full FP4 range, while a scale of 4 spreads the error more evenly across the block. Use it to quantize a model to FP4 while it loads, or to keep training a model with FP4 weights.

The table below summarizes what Four Over Six supports in Transformers.

| | |
|---|---|
| Config | [`FourOverSixConfig`] |
| `quant_method` | `fouroversix` |
| Workflow | Quantize while loading |
| Bits | 4 |
| Hardware | CUDA (fastest on NVIDIA Blackwell), with a slower fallback on other devices |
| Save | ✅ |
| Fine-tune | Only with `keep_master_weights=True` |
| Auto-compile | ❌ |

## Install Four Over Six

Install the fouroversix package without build isolation.

```bash
pip install fouroversix --no-build-isolation
```

The install builds CUDA kernels that need a Blackwell GPU. On other hardware, set `SKIP_CUDA_BUILD=1` to install only the slower reference implementation.

```bash
SKIP_CUDA_BUILD=1 pip install fouroversix --no-build-isolation
```

## Quantize a model with Four Over Six

Create a [`FourOverSixConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. The default config quantizes every linear layer except `lm_head` to NVFP4.

```py
from transformers import AutoModelForCausalLM, FourOverSixConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=FourOverSixConfig(),
    device_map="cuda",
)
```

## Configure Four Over Six

These [`FourOverSixConfig`] parameters change how the weights and activations are quantized. Most of them also have `weight_`, `activation_`, and `gradient_` variants that override the setting for one kind of tensor.

| Parameter | Default | Effect |
|---|---|---|
| `dtype` | `"nvfp4"` | FP4 format for weights, activations, and gradients, either `"nvfp4"` or `"mxfp4"`. |
| `scale_rule` | `"mse"` | How each block's scale is chosen. `"mse"`, `"mae"`, and `"abs_max"` pick between 4 and 6 by that error measure. `"static_6"` is standard NVFP4 quantization, and `"static_4"` scales every block to 4. |
| `keep_master_weights` | `False` | Keeps the high-precision weights and quantizes them in every forward pass. Required for training. |
| `weight_scale_2d` | `False` | Computes weight scales over 2D blocks instead of 1D blocks. Use it for training. |
| `matmul_backend` | `None` | Matmul backend, either `"cutlass"` or `"pytorch"`. With `None`, CUTLASS is used when it's available. |
| `module_config_overrides` | `None` | Maps module names to dictionaries of quantization options, so individual layers can use different settings. |

See [`FourOverSixConfig`] for the rest of the parameters.

## Train with quantized weights

The Four Over Six linear layer has a backward pass, so you can keep training a quantized model. Set `keep_master_weights=True` so the high-precision weights stay in memory and are re-quantized in each forward pass, and set `weight_scale_2d=True` as the config recommends for training.

The Four Over Six authors recommend leaving the last few layers of the network in high precision. The example below also excludes the attention projections of decoder layer 34. Add more layers, like `self_attn.o_proj` and the MLP projections, as needed.

```py
from transformers import AutoModelForCausalLM, FourOverSixConfig

quantization_config = FourOverSixConfig(
    keep_master_weights=True,
    weight_scale_2d=True,
    modules_to_not_convert=[
        "lm_head",
        "model.layers.34.self_attn.q_proj",
        "model.layers.34.self_attn.k_proj",
        "model.layers.34.self_attn.v_proj",
    ],
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

## Limitations

- Without `keep_master_weights=True`, the quantized model can't be fine-tuned.
- On hardware other than NVIDIA Blackwell GPUs, Four Over Six simulates FP4 with FP32 matmuls, so you get the accuracy loss of FP4 without the speedup.

## Resources

- [Four Over Six](https://arxiv.org/abs/2512.02010) preprint
- [fouroversix](https://github.com/mit-han-lab/fouroversix) repository

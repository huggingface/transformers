<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# SINQ

[Sinkhorn-Normalized Quantization (SINQ)](https://github.com/huawei-csl/SINQ) balances the scale of each weight matrix's rows and columns before quantizing it, which reduces the error from outliers without calibration data. Use it as a calibration-free alternative to [HQQ](./hqq) for 2- to 8-bit weights on a single GPU.

The table below summarizes what SINQ supports in Transformers.

| | |
|---|---|
| Config | [`SinqConfig`] |
| `quant_method` | `sinq` |
| Workflow | Quantize while loading |
| Bits | 2, 3, 4, 5, 6, 8 |
| Hardware | One CUDA GPU, CPU |
| Save | ✅ |
| Fine-tune | ✅ |
| Auto-compile | ❌ |

## Install SINQ

Install the sinq package.

```bash
pip install sinq
```

## Quantize a model with SINQ

Create a [`SinqConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. The example below quantizes a model to 4 bits and keeps `lm_head` in bf16.

```py
from transformers import AutoModelForCausalLM, SinqConfig

quantization_config = SinqConfig(nbits=4, group_size=64, modules_to_not_convert=["lm_head"])
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    quantization_config=quantization_config,
    device_map="cuda:0",
)
```

Without a CUDA GPU, SINQ quantizes and runs the model on the CPU, which is much slower.

## Configure SINQ

These [`SinqConfig`] parameters change how the weights are quantized.

| Parameter | Default | Effect |
|---|---|---|
| `nbits` | `4` | Number of bits per weight. |
| `group_size` | `64` | Number of weights that share a scale. Use a multiple of 8. |
| `tiling_mode` | `"1D"` | How the weight matrix is split into tiles, either `"1D"` or `"2D"`. |
| `modules_to_not_convert` | `None` | Modules to leave unquantized. With the default, every linear layer is quantized, including `lm_head`. |

## Limitations

- SINQ runs on one device. A `device_map` that spreads the model across several devices, including CPU offloading, raises a `RuntimeError`.
- Transformers supports only the calibration-free `method="sinq"`. Quantizing with the calibrated `method="asinq"` raises a `ValueError`. Use the [SINQ](https://github.com/huawei-csl/SINQ) library to quantize with A-SINQ.

## Resources

- [SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights](https://arxiv.org/abs/2509.22944) paper
- [SINQ](https://github.com/huawei-csl/SINQ) repository

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

# Metal

Metal quantization packs weights into `uint32` tensors with an affine scale and bias for each group, and runs a fused dequantize and matmul kernel on the Apple silicon GPU. The kernels come from [MLX](https://github.com/ml-explore/mlx) and load from the Hub ([kernels-community/mlx-quantization-metal-kernels](https://huggingface.co/kernels-community/mlx-quantization-metal-kernels)) through the [kernels](https://github.com/huggingface/kernels) library. Use it to run 2-, 4-, or 8-bit models on a Mac with MPS.

The table below summarizes what Metal supports in Transformers.

| | |
|---|---|
| Config | [`MetalConfig`] |
| `quant_method` | `metal` |
| Workflow | Quantize while loading |
| Bits | 2, 4, 8 |
| Hardware | Apple silicon (MPS) |
| Save | ✅ |
| Fine-tune | ❌ |
| Auto-compile | ❌ |

## Install Metal

Install Transformers with the kernels extra. The Metal kernels download from the Hub the first time a quantized layer runs.

```bash
pip install -U "transformers[kernels]"
```

## Quantize a model with Metal

Create a [`MetalConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. Set `device_map="mps"` so the model runs on the Apple silicon GPU. The example below quantizes every linear layer to 4 bits in groups of 64 weights.

```py
from transformers import AutoModelForCausalLM, MetalConfig

quantization_config = MetalConfig(bits=4, group_size=64)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=quantization_config,
    device_map="mps",
)
```

A checkpoint saved with Metal quantization stores the packed weights, scales, and biases. Loading it with [`~PreTrainedModel.from_pretrained`] reads them directly without quantizing again.

## Configure Metal

These [`MetalConfig`] parameters change how the weights are quantized.

| Parameter | Default | Effect |
|---|---|---|
| `bits` | `4` | Number of bits per weight. Supports 2, 4, and 8. |
| `group_size` | `64` | Number of weights that share a scale and bias. Must be positive. |
| `modules_to_not_convert` | `None` | Modules to leave unquantized. |

## Load a Metal checkpoint without Apple silicon

If you load a Metal checkpoint on a machine without MPS, Transformers logs a warning and dequantizes the weights back to floating point, so the model runs as a regular model on the CPU or a CUDA GPU. To dequantize on any machine, set `dequantize=True`. The example below loads the model from [Quantize a model with Metal](#quantize-a-model-with-metal) after it was saved with `model.save_pretrained("Qwen3-0.6B-metal")`.

```py
from transformers import AutoModelForCausalLM, MetalConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen3-0.6B-metal",
    quantization_config=MetalConfig(dequantize=True),
    device_map="cpu",
)
```

`dequantize` only applies when loading. It isn't saved in the checkpoint's config.

## Limitations

- Quantizing a model needs an MPS device. Quantizing without one raises a `RuntimeError`.
- Loading without the kernels package installed raises an `ImportError`.
- Quantizing a model with `"cpu"` or `"disk"` in its `device_map` raises a `ValueError`.

## Resources

- [MLX](https://github.com/ml-explore/mlx) repository
- [kernels-community/mlx-quantization-metal-kernels](https://huggingface.co/kernels-community/mlx-quantization-metal-kernels) on the Hub

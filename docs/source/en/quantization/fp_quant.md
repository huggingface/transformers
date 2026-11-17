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


# FP-Quant

[FP-Quant](https://github.com/IST-DASLab/FP-Quant) quantizes weights and activations to the [MXFP4](./mxfp4) or [NVFP4](./nvfp4) formats, applying a rotation (a Hadamard transform by default) before quantizing, and runs the matmuls with [QuTLASS](https://github.com/IST-DASLab/qutlass) kernels on NVIDIA Blackwell GPUs. Use it for FP4 inference on Blackwell GPUs, or emulate FP4 quantization on other GPUs to evaluate accuracy or run quantization-aware training (QAT).

The table below summarizes what FP-Quant supports in Transformers.

| | |
|---|---|
| Config | [`FPQuantConfig`] |
| `quant_method` | `fp_quant` |
| Workflow | Quantize while loading |
| Bits | 4 |
| Hardware | NVIDIA Blackwell GPUs; other CUDA GPUs and Intel XPU with `pseudoquantization=True` |
| Save | ✅ |
| Fine-tune | Only with `store_master_weights=True` |
| Auto-compile | ❌ |

## Install FP-Quant

Install fp_quant 0.3.2 or later. To run the real FP4 kernels on a Blackwell GPU, also build QuTLASS 0.2.0 or later from source.

```bash
pip install -U "fp_quant>=0.3.2"
git clone https://github.com/IST-DASLab/qutlass.git
cd qutlass && pip install --no-build-isolation .
```

## Quantize a model with FP-Quant

Create a [`FPQuantConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. Set `device_map` to your GPUs. The default config quantizes every linear layer except `lm_head` to NVFP4.

```py
from transformers import AutoModelForCausalLM, FPQuantConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=FPQuantConfig(),
    device_map="cuda",
)
```

FP-Quant only supports bf16, so the model always loads in bf16. Any other `dtype` is replaced with a warning.

Without a Blackwell GPU, set `pseudoquantization=True`. FP-Quant then emulates FP4 quantization with Triton kernels. It reproduces the accuracy of the quantized model but doesn't speed it up.

```py
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=FPQuantConfig(forward_dtype="mxfp4", pseudoquantization=True),
    device_map="cuda",
)
```

To load a model that was quantized ahead of time with GPTQ for better accuracy, pass its Hub repository to [`~PreTrainedModel.from_pretrained`] without a `quantization_config`. The [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/models?search=FPQuant) organization hosts FP-Quant checkpoints, and the [FP-Quant](https://github.com/IST-DASLab/FP-Quant) repository has the code to create them.

## Configure FP-Quant

These [`FPQuantConfig`] parameters change how the weights and activations are quantized.

| Parameter | Default | Effect |
|---|---|---|
| `forward_dtype` | `"nvfp4"` | FP4 format for the forward pass, either `"nvfp4"` or `"mxfp4"`. |
| `hadamard_group_size` | `None` | Size of the Hadamard transform. With `None`, it's 16 for NVFP4 and 32 for MXFP4. NVFP4 accepts 16, 32, 64, and 128, and MXFP4 accepts 32, 64, and 128. |
| `transform_init` | `"hadamard"` | How the transform matrix is initialized, either `"hadamard"`, `"identity"`, or `"gsr"`. |
| `pseudoquantization` | `False` | Emulates FP4 with Triton kernels instead of running the QuTLASS kernels. Required on GPUs other than Blackwell. |

See [`FPQuantConfig`] for the rest of the parameters.

## Train with quantized weights

FP-Quant supports QAT. Set `store_master_weights=True` to keep the high-precision weights so they can be updated, and set `forward_method="quest"`, which works better than `"abs_max"` for QAT. The `"quest"` method only works with MXFP4. With an MXFP4 forward pass, `backward_dtype` can also quantize the backward pass to `"mxfp8"` or `"mxfp4"` instead of `"bf16"`.

```py
from transformers import AutoModelForCausalLM, FPQuantConfig

quantization_config = FPQuantConfig(
    forward_dtype="mxfp4",
    forward_method="quest",
    store_master_weights=True,
    pseudoquantization=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

The FP-Quant authors note that QAT currently only works with `pseudoquantization=True`.

## Limitations

- FP-Quant needs a CUDA GPU or Intel XPU. Loading without one raises a `NotImplementedError`.
- Without QuTLASS installed, loading with `pseudoquantization=False` raises an `ImportError`.
- NVFP4 pseudo-quantization needs a CUDA GPU with compute capability 9.0 or higher (Hopper or newer). On older GPUs, it raises a `ValueError`. Use `forward_dtype="mxfp4"` instead.
- Without `pseudoquantization`, loading without a `device_map` raises a `ValueError`. A `device_map` that spreads the model across a GPU and the CPU raises a `ValueError` too.
- A `device_map` that includes `"disk"` raises a `ValueError`.
- Pseudo-quantization doesn't support multiple GPUs.
- Training without `store_master_weights=True` isn't supported.

## Resources

- [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202) paper
- [FP-Quant](https://github.com/IST-DASLab/FP-Quant) repository
- [QuTLASS](https://github.com/IST-DASLab/qutlass) repository, including speedup benchmarks

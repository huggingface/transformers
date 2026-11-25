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

# FBGEMM

[FBGEMM (Facebook GEneral Matrix Multiplication)](https://github.com/pytorch/FBGEMM) FP8 quantization stores weights in FP8 with one scale per output channel, and quantizes activations to FP8 with one scale per token at runtime (W8A8). The matmul runs on FBGEMM's row-wise FP8 kernel. Use it for FP8 weight and activation quantization on Hopper or newer NVIDIA GPUs, or on Intel XPUs, without calibration data.

The table below summarizes what FBGEMM supports in Transformers.

| | |
|---|---|
| Config | [`FbgemmFp8Config`] |
| `quant_method` | `fbgemm_fp8` |
| Workflow | Quantize while loading |
| Bits | 8 |
| Hardware | CUDA (compute capability 9.0+), XPU |
| Save | ✅ |
| Fine-tune | ❌ |
| Auto-compile | ❌ |

## Install FBGEMM

On a CUDA GPU, install the fbgemm-gpu and [Accelerate](https://huggingface.co/docs/accelerate/index) libraries. If you have installation issues, try the fbgemm-gpu [nightly release](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/development/InstallationInstructions.html#fbgemm-gpu-install-libraries).

```bash
pip install -U fbgemm-gpu accelerate
```

On an XPU, Transformers downloads the FP8 quantization kernel from the [kernels-community/fp8-fbgemm](https://huggingface.co/kernels-community/fp8-fbgemm) repository on the Hub instead. Install the [kernels](https://huggingface.co/docs/kernels/index) and Accelerate libraries.

```bash
pip install -U "transformers[kernels]" accelerate
```

## Quantize a model with FBGEMM

Create a [`FbgemmFp8Config`] and pass it to [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModelForCausalLM, FbgemmFp8Config

quantization_config = FbgemmFp8Config()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

FBGEMM always loads the model in bf16. If you pass a different `dtype`, Transformers logs a warning and switches it to `torch.bfloat16`.

`activation_scale_ub` caps the per-token activation scale when activations are quantized at runtime. It defaults to `1200.0`, and you can change it when you load a pre-quantized checkpoint.

## Limitations

- FBGEMM needs a CUDA GPU or an XPU. Loading without one raises an `ImportError`.
- On CUDA, FBGEMM needs compute capability 9.0 or higher, like an H100. An older GPU raises a `ValueError`.
- Quantizing while loading can't offload to the CPU or disk. A `device_map` that includes `"cpu"` or `"disk"` raises a `ValueError`. A pre-quantized checkpoint can offload.
- FBGEMM quantizes `nn.Linear` layers and Llama 4 text experts. Fused expert modules in other mixture-of-experts models stay unquantized.
- FBGEMM-quantized models can't be fine-tuned.

## Resources

- [FBGEMM](https://github.com/pytorch/FBGEMM) repository
- [Open-sourcing FBGEMM for state-of-the-art server-side inference](https://engineering.fb.com/2018/11/07/ml-applications/fbgemm/) blog post

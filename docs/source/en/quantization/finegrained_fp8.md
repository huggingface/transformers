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

# Fine-grained FP8

Fine-grained FP8 stores weights in FP8 (E4M3) with one scale per 2D block of weights, 128x128 by default, and quantizes activations to FP8 at runtime with one scale per token for each 128-value group of input channels. It's the format [DeepSeek-V3](https://hf.co/papers/2412.19437) and many other FP8 checkpoints on the Hub use. Use it to run FP8 checkpoints, or to quantize a model to FP8 while it loads, on Ada, Hopper, or Blackwell GPUs and on Intel XPUs.

The table below summarizes what Fine-grained FP8 supports in Transformers.

| | |
|---|---|
| Config | [`FineGrainedFP8Config`] |
| `quant_method` | `fp8`, `mxfp8` |
| Workflow | Quantize while loading, load pre-quantized |
| Bits | 8 |
| Hardware | CUDA (compute capability 8.9+), XPU |
| Save | ✅ |
| Fine-tune | ✅ |
| Auto-compile | ✅ |

## Install the Fine-grained FP8 kernels

Transformers downloads the FP8 matmul kernel from the [kernels-community/finegrained-fp8](https://huggingface.co/kernels-community/finegrained-fp8) repository on the Hub. Install the [kernels](https://huggingface.co/docs/kernels/index) and [Accelerate](https://huggingface.co/docs/accelerate/index) libraries.

```bash
pip install -U "transformers[kernels]" accelerate
```

## Quantize a model with Fine-grained FP8

Create a [`FineGrainedFP8Config`] and pass it to [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModelForCausalLM, FineGrainedFP8Config

quantization_config = FineGrainedFP8Config()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

## Configure Fine-grained FP8

These [`FineGrainedFP8Config`] parameters change how the weights and activations are quantized.

| Parameter | Default | Effect |
|---|---|---|
| `weight_block_size` | `(128, 128)` | Size of the weight block that shares one scale. Set it to `None` for one scale per weight tensor. |
| `activation_scheme` | `"dynamic"` | `"dynamic"` computes activation scales at runtime. `"static"` uses one stored activation scale per layer, read from a pre-quantized checkpoint. `"none"` keeps activations unquantized (weight-only FP8): the weights are dequantized to the activations' dtype in forward. |
| `scale_fmt` | `"float"` | Storage format of the weight scales. `"float"` stores float32 scales. `"ue8m0"` stores 1-byte power-of-two scales (`torch.float8_e8m0fnu`), as in DeepSeek-V4 checkpoints, and needs PyTorch 2.7 or later. |
| `modules_to_convert` | `None` | Embedding tables to load in FP8 from a pre-quantized checkpoint, matched by module name suffix. |

## Load a pre-quantized FP8 checkpoint

Checkpoints quantized with Fine-grained FP8 carry their quantization config, so load them without a [`FineGrainedFP8Config`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-FP8", device_map="cuda")
```

Fine-grained FP8 also loads MXFP8 checkpoints, whose config sets `quant_method="mxfp8"`. MXFP8 stores FP8 weights with one power-of-two scale, saved as a byte, per block of 32 weights.

Checkpoints that name their scales differently load too: DeepSeek-V4's `scale`, Mistral Large 4's `qscale_weight`, and
per-tensor FP8's `weight_scale` / `input_scale`.

Checkpoints that mix FP8 with other methods per layer load as a [quantization plan](./overview#quantization-plans),
with fine-grained FP8 for their FP8 layers:

- [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is fine-grained FP8 (ue8m0 scales) except for its
  routed experts, saved as MXFP4 (`expert_dtype: "fp4"`), which load with [MXFP4](./mxfp4).
- ModelOpt checkpoints with `quant_algo: "MIXED_PRECISION"`, e.g. [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4),
  list a `quant_algo` per layer: `FP8` (per-tensor, static activations) and `FP8_PB_WO` (`(128, 128)` blocks,
  weight-only) layers load with fine-grained FP8, `NVFP4` ones with [NVFP4](./nvfp4).

## Use the DeepGEMM fast path

On Hopper (SM90) and Blackwell (SM100) GPUs, FP8 linear layers dispatch to the [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels from [kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm) when `weight_block_size=(128, 128)` and `activation_scheme="dynamic"`. DeepGEMM is faster than the default Triton kernel on the shapes it supports. The `kernels` install above enables it.

DeepGEMM compiles its kernels with the system `nvcc` the first time it runs, so you need a full CUDA toolkit, not a runtime-only install. Hopper needs CUDA 12.3 or later and Blackwell needs CUDA 12.9 or later. Transformers looks for the toolkit in `CUDA_HOME`, then `CUDA_PATH`, then the directory of `nvcc` on your `PATH`, then `/usr/local/cuda`. Set `CUDA_HOME` to the toolkit root if Transformers doesn't find it.

If DeepGEMM can't load or declines a layer, Transformers logs a warning and falls back to the Triton kernel. Set `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` to always use the Triton kernel for linear layers.

On Blackwell, DeepGEMM only reads UE8M0 scales. Linear layers with float32 scales (`scale_fmt="float"`) use the Triton kernel instead.

For mixture-of-experts layers, DeepGEMM is opt-in. Pass `experts_implementation="deepgemm"` to [`~PreTrainedModel.from_pretrained`], or `"deepgemm_megamoe"` on Blackwell. See the [Experts backends](../experts_interface) guide for every option.

## Limitations

- Quantizing while loading needs a CUDA GPU or an XPU. Without one, loading raises a `RuntimeError`. A pre-quantized checkpoint loads without an accelerator, but Transformers dequantizes it to bf16 and logs a warning.
- On CUDA, FP8 compute needs compute capability 8.9 or higher, like an RTX 4090 or H100. On an older GPU, Transformers dequantizes the model to bf16 and logs a warning.
- Quantizing while loading can't offload to the CPU or disk. A `device_map` that includes `"cpu"` or `"disk"` alongside other devices raises a `ValueError`. A pre-quantized checkpoint can offload.
- On Blackwell, the `"deepgemm"` experts backend raises a `NotImplementedError` for float32 scales. Use a `scale_fmt="ue8m0"` checkpoint, or the `"grouped_mm"` or `"batched_mm"` experts backends.
- The `"grouped_mm"` and `"batched_mm"` experts backends don't support `activation_scheme="static"` and raise a `NotImplementedError`.
- `scale_fmt="ue8m0"` needs PyTorch 2.7 or later. Older versions raise a `RuntimeError`.

## Resources

- [DeepSeek-V3 Technical Report](https://hf.co/papers/2412.19437) paper
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) repository
